import os
import argparse
import pandas as pd
import xgboost as xgb

def main(args):
    # Get the training data directory from the SM_CHANNEL_TRAIN environment variable.
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir:
        raise ValueError("SM_CHANNEL_TRAIN environment variable is not set.")
    
    print("SM_CHANNEL_TRAIN directory:", train_dir)
    print("Files in SM_CHANNEL_TRAIN:", os.listdir(train_dir))
    
    # Build paths to your CSV files.
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")
    
    print("Loading training data from:", X_train_path)
    train_data = pd.read_csv(X_train_path, header=None)
    
    print("Loading training labels from:", y_train_path)
    train_labels = pd.read_csv(y_train_path, header=None)
    
    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)
    
    # Create a DMatrix for XGBoost.
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    
    # Define the XGBoost parameters.
    params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric
    }
    
    print("Training with parameters:", params)
    bst = xgb.train(params, dtrain, num_boost_round=args.num_round)
    
    # Save the model to the location specified by SM_MODEL_DIR.
    model_dir = os.environ.get("SM_MODEL_DIR")
    if not model_dir:
        raise ValueError("SM_MODEL_DIR environment variable is not set.")
    
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    bst.save_model(model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth for base learners.")
    parser.add_argument("--eta", type=float, default=0.1, help="Boosting learning rate.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of training instances.")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Subsample ratio of columns when constructing each tree.")
    parser.add_argument("--eval_metric", type=str, default="rmse", help="Evaluation metric.")
    parser.add_argument("--num_round", type=int, default=100, help="Number of boosting rounds.")
    
    args = parser.parse_args()
    main(args)

























import sagemaker
from sagemaker.xgboost.estimator import XGBoost

# Initialize SageMaker session.
sagemaker_session = sagemaker.Session()
bucket = "ml-models-bucket-eto1"
role = "arn:aws:iam::825765427114:role/service-role/AmazonSageMaker-ExecutionRole-20250210T121816"

# Upload the current directory. Ensure that X_train.csv and y_train.csv are present.
training_data_s3 = sagemaker_session.upload_data(
    path=".",               # Current directory containing train.py, X_train.csv, y_train.csv
    bucket=bucket,
    key_prefix="data"       # S3 prefix: s3://ml-models-bucket-eto1/data/
)

# Configure the XGBoost estimator in Script Mode.
xgb_estimator = XGBoost(
    entry_point="train.py",  # Our unified training script.
    source_dir=".",          # Directory containing train.py.
    framework_version="1.5-1",  # Use an appropriate XGBoost version.
    instance_type="ml.c4.2xlarge",
    instance_count=1,
    role=role,
    sagemaker_session=sagemaker_session,
    use_spot_instances=True,  # For cost efficiency.
    max_run=3600,             # Maximum training runtime in seconds.
    max_wait=7200             # Maximum wait time for spot instances in seconds.
)

# Launch the training job with one channel named "train".
xgb_estimator.fit({"train": training_data_s3})
print("✅ Training job submitted.")


#second script 

