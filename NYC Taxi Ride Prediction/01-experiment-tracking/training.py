import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow # for tracking the experiment

# we not gonna set the tracking_url because we gonna save everything to the local system
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#set the tracking uri - needed for the ssqlite 
mlflow.set_tracking_uri("sqlite:///backend.db")
#set experiment
mlflow.set_experiment("random-forest-experiment-1")
#enable auto logging so that everything can be automatically loggeed to MLflow 
mlflow.sklearn.autolog()

# load the data save in the filename/path
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    # specify the run so the experiment knows where the run is starting  
    with mlflow.start_run():
        # load data for train and validation
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
        # instantiate a model for the  training and prediction
        
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        # the bottom part of this code will not be logged automatically by the auto log, hence we need to log
        # it manually

        y_pred = rf.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        # this part is optional
        mlflow.log_metric("rmse", rmse)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)


