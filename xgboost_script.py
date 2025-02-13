import os
import argparse
import pandas as pd
import xgboost as xgb

def main():
    # Get the training data directory from the environment variable.
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir:
        raise ValueError("SM_CHANNEL_TRAIN environment variable is not set.")
    
    print("SM_CHANNEL_TRAIN directory:", train_dir)
    print("Files in SM_CHANNEL_TRAIN:", os.listdir(train_dir))
    
    # Build file paths for training data and labels.
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")
    
    print("Loading training data from:", X_train_path)
    train_data = pd.read_csv(X_train_path, header=None)
    
    print("Loading training labels from:", y_train_path)
    train_labels = pd.read_csv(y_train_path, header=None)
    
    print("Shape of training data:", train_data.shape)
    print("Shape of training labels:", train_labels.shape)
    
    # Create DMatrix for XGBoost.
    dtrain = xgb.DMatrix(train_data, label=train_labels)
    
    # Parse hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth for base learners.")
    parser.add_argument("--eta", type=float, default=0.1, help="Boosting learning rate.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio of the training instances.")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Subsample ratio of columns when constructing each tree.")
    parser.add_argument("--eval_metric", type=str, default="rmse", help="Evaluation metric for validation data.")
    parser.add_argument("--num_round", type=int, default=100, help="Number of boosting rounds.")
    parser.add_argument("--objective", type=str, default="reg:squarederror", help="Learning objective.")
    args = parser.parse_args()
    
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "objective": args.objective
    }
    
    print("Training with parameters:", params)
    bst = xgb.train(params, dtrain, num_boost_round=args.num_round)
    
    # Save the model to the directory specified by SM_MODEL_DIR.
    model_dir = os.environ.get("SM_MODEL_DIR")
    if not model_dir:
        raise ValueError("SM_MODEL_DIR environment variable is not set.")
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    bst.save_model(model_path)
    print("Model saved to:", model_path)

if __name__ == "__main__":
    main()
