import boto3

sagemaker_client = boto3.client("sagemaker")
endpoint_name = "cibil-score-predictor"

response = sagemaker_client.create_model(
    ModelName=endpoint_name,
    ExecutionRoleArn="arn:aws:iam::354918375730:role/service-role/AmazonSageMaker-ExecutionRole-20250122T184441",
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
        "InstanceType": "ml.c4.2xlarge",
        "InitialInstanceCount": 1
    }]
)

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=f"{endpoint_name}-config"
)

print("✅ Model deployed successfully as a SageMaker Endpoint!")
