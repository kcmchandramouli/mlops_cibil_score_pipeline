import sagemaker
from sagemaker.xgboost.estimator import XGBoost

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
bucket = "ml-models-bucket-eto1"
role = "arn:aws:iam::825765427114:role/service-role/AmazonSageMaker-ExecutionRole-20250210T121816"

# IMPORTANT:
# Make sure X_train.csv and y_train.csv are both in the same local folder as train.py.
# We'll upload the entire folder (.) so that S3 has both files under "data/".
training_data_s3 = sagemaker_session.upload_data(
    path=".",                # This folder must contain X_train.csv and y_train.csv
    bucket=bucket,
    key_prefix="data"        # S3 prefix: s3://ml-models-bucket-eto1/data/
)

# Configure the XGBoost estimator (Script Mode)
xgb = XGBoost(
    entry_point="xgboost_script.py",  # This script must read X_train.csv and y_train.csv
    source_dir=".",                   # The directory where xgboost_script.py is located
    framework_version="1.5-1",        # Update to a newer version (1.5-1 or above)
    instance_type="ml.c4.2xlarge",
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    use_spot_instances=True,          # Enable spot instances for cost efficiency
    max_run=3600,                     # Max training job runtime (1 hour)
    max_wait=7200                     # Max wait time for spot instance (2 hours)
)

# Start the training job
# We pass a single channel named "train", which has BOTH X_train.csv and y_train.csv
xgb.fit({"train": training_data_s3})

print("✅ Model trained successfully and saved to S3!")
