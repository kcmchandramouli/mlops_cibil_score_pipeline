import sagemaker
from sagemaker.xgboost.estimator import XGBoost

sagemaker_session = sagemaker.Session()
bucket = "ml-models-bucket-eto1"
role = "arn:aws:iam::825765427114:role/service-role/AmazonSageMaker-ExecutionRole-20250210T121816"

X_train_s3 = sagemaker_session.upload_data(path="X_train.csv", bucket=bucket, key_prefix="data")
y_train_s3 = sagemaker_session.upload_data(path="y_train.csv", bucket=bucket, key_prefix="data")

xgb = XGBoost(
    entry_point="xgboost_script.py",
    framework_version="1.3-1",
    instance_type="ml.c4.2xlarge",  # Changed to ml.c4.2xlarge
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    use_spot_instances=True  # Use Spot Instances to stay within quota
)

xgb.fit({"train": X_train_s3})
print("✅ Model trained successfully and saved to S3!")
