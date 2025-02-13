import os
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    # 1) Read environment variables for the train channel
    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")

    # 2) Load the CSV data
    print("Reading training data from:", X_train_path)
    print("Reading training labels from:", y_train_path)
    train_data = pd.read_csv(X_train_path, header=None)
    train_labels = pd.read_csv(y_train_path, header=None)

    dtrain = xgb.DMatrix(train_data, label=train_labels)

    # 3) Define parameters
    params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }

    # 4) Train the model
    print("Training XGBoost...")
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # 5) Save the model
    model_dir = os.environ["SM_MODEL_DIR"]
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    bst.save_model(model_path)
    print(f"Model saved to {model_path}")
