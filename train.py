import sagemaker
from sagemaker.xgboost.estimator import XGBoost

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
bucket = "ml-models-bucket-eto1"
role = "arn:aws:iam::825765427114:role/service-role/AmazonSageMaker-ExecutionRole-20250210T121816"

# Upload the current directory so that both X_train.csv and y_train.csv are uploaded.
# Make sure X_train.csv and y_train.csv are in the same folder as train.py.
training_data_s3 = sagemaker_session.upload_data(
    path=".",                # Current folder must include both CSV files.
    bucket=bucket,
    key_prefix="data"        # S3 prefix, e.g., s3://ml-models-bucket-eto1/data/
)

# Configure the XGBoost estimator (Script Mode)
xgb = XGBoost(
    entry_point="xgboost_script.py",  # Our training script.
    source_dir=".",                   # The folder containing the script.
    framework_version="1.5-1",        # Use a recent framework version.
    instance_type="ml.c4.2xlarge",    # Change instance type as needed.
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    use_spot_instances=True,          # For cost efficiency.
    max_run=3600,                     # Maximum training runtime (seconds).
    max_wait=7200                     # Maximum wait time for spot instances (seconds).
)

# Start the training job with one channel named "train"
xgb.fit({"train": training_data_s3})
print("✅ Model trained successfully and saved to S3!")
