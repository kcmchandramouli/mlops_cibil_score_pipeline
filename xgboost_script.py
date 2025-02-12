import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
import xgboost as xgb
import os
import pandas as pd
import numpy as np

# Load the dataset
train_data = pd.read_csv(os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'X_train.csv'))
train_labels = pd.read_csv(os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'y_train.csv'))

# Prepare the training data
dtrain = xgb.DMatrix(train_data, label=train_labels)

# Define the parameters for the XGBoost model
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse'
}

# Train the XGBoost model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Save the model to the output directory
output_dir = os.environ['SM_OUTPUT_DIR']
bst.save_model(os.path.join(output_dir, 'xgboost_model.bin'))
