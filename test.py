import boto3
import json
import numpy as np

test_data = np.array([[45, 100000, 15000, 3, 80, 0.2]])

runtime = boto3.client("sagemaker-runtime")
endpoint_name = "cibil-score-predictor"

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=json.dumps(test_data.tolist())
)

prediction = response["Body"].read().decode("utf-8")
print(f"Predicted CIBIL Score: {prediction}")
