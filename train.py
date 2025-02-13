import sagemaker
from sagemaker.xgboost.estimator import XGBoost

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
bucket = "ml-models-bucket-eto1"
role = "arn:aws:iam::825765427114:role/service-role/AmazonSageMaker-ExecutionRole-20250210T121816"

# Upload training data to S3
X_train_s3 = sagemaker_session.upload_data(path="X_train.csv", bucket=bucket, key_prefix="data")

# Configure the XGBoost estimator
xgb = XGBoost(
    entry_point="xgboost_script.py",
    framework_version="1.3-1",
    instance_type="ml.c4.2xlarge",  # Using ml.c4.2xlarge as per available quota
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    use_spot_instances=True,   # Enable spot instances for cost efficiency
    max_run=3600,              # Maximum training job runtime (1 hour)
    max_wait=7200              # Maximum wait time for spot instance (2 hours)
)

# Start the training job
xgb.fit({"train": X_train_s3})
print("✅ Model trained successfully and saved to S3!")
