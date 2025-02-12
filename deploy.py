import boto3

sagemaker_client = boto3.client("sagemaker")
endpoint_name = "cibil-score-predictor"

response = sagemaker_client.create_model(
    ModelName=endpoint_name,
    ExecutionRoleArn="arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole",
    PrimaryContainer={
        "Image": "433757028032.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.3-1",
        "ModelDataUrl": "s3://ml-models-bucket/model.tar.gz"
    }
)

sagemaker_client.create_endpoint_config(
    EndpointConfigName=f"{endpoint_name}-config",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": endpoint_name,
        "InstanceType": "ml.m5.large",
        "InitialInstanceCount": 1
    }]
)

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=f"{endpoint_name}-config"
)

print("✅ Model deployed successfully as a SageMaker Endpoint!")
