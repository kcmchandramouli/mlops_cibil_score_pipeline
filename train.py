import os
import sys
import argparse
import pandas as pd
import xgboost as xgb

def run_training():
    """
    This function runs INSIDE the SageMaker training container.
    It expects SM_CHANNEL_TRAIN and SM_MODEL_DIR to be set.
    """
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir:
        print("ERROR: SM_CHANNEL_TRAIN is not set. Are we inside SageMaker?")
        sys.exit(1)

    print("Inside container. SM_CHANNEL_TRAIN =", train_dir)
    print("Files in train_dir:", os.listdir(train_dir))

    # Paths to training files
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")

    print("Loading training data from:", X_train_path)
    train_data = pd.read_csv(X_train_path, header=None)

    print("Loading training labels from:", y_train_path)
    train_labels = pd.read_csv(y_train_path, header=None)

    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)

    # Parse hyperparameters passed from SageMaker
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--eval_metric", type=str, default="rmse")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    args = parser.parse_args()

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "objective": args.objective
    }

    print("XGBoost parameters:", params)

    # Train XGBoost
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    bst = xgb.train(params, dtrain, num_boost_round=args.num_round)

    # Save the model to SM_MODEL_DIR
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    bst.save_model(model_path)
    print("Model saved to:", model_path)

def launch_training_job():
    """
    This function runs LOCALLY (outside SageMaker) to create the training job.
    It uploads local files to S3 and starts the job using this same script.
    """
    import sagemaker
    from sagemaker.xgboost.estimator import XGBoost

    print("No SM_CHANNEL_TRAIN variable => Running locally to launch SageMaker job.")

    sagemaker_session = sagemaker.Session()
    role = "arn:aws:iam::183631350288:role/service-role/AmazonSageMaker-ExecutionRole-20250213T183561"
    bucket = "cibils3"

    # Upload the current folder so that X_train.csv and y_train.csv go to S3
    training_data_s3 = sagemaker_session.upload_data(
        path=".",     # Must contain X_train.csv and y_train.csv
        bucket=bucket,
        key_prefix="data"
    )

    # Create an XGBoost Estimator that points back to this same script
    xgb_estimator = XGBoost(
        entry_point="train.py",      # This script is the entry point
        source_dir=".",              # Directory containing this script
        framework_version="1.5-1",   # Use a recent XGBoost version
        instance_type="ml.c4.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_run=3600,
        max_wait=7200
    )

    # Launch training job with channel name "train"
    xgb_estimator.fit({"train": training_data_s3})
    print("✅ SageMaker training job started successfully!")

def main():
    # If SM_CHANNEL_TRAIN is set, we're inside the container => run training
    if "SM_CHANNEL_TRAIN" in os.environ:
        run_training()
    else:
        # Otherwise, we're local => launch the SageMaker job
        launch_training_job()

if __name__ == "__main__":
    main()
