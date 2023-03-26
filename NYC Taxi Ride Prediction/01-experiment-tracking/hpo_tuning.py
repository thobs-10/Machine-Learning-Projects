import argparse
import os
import pickle

import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# running sqlite for backend storage
# mlflow ui --backend-store-uri sqlite:///mlflow.db( you can customize the name of the db to the one you prefer)
mlflow.set_tracking_uri("http://127.0.0.1:5000") # the url to visualize the experiments to the local host tracking server
# keep in mind this url for the server is still running and the server is running also in the backend store
mlflow.set_experiment("random-forest-hyperopt-1") # experiment name

# for loading  datasets
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# the run function  is responsible for beginning the experiment run
def run(data_path, num_trials):
    # load dataset for train and validation
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    def objective(params):
        # an inner function to track the runs and record the parameters
        with mlflow.start_run():
            # log the parameters that are past in arguments
            mlflow.log_params(params)
            mlflow.set_tag("model-name","random-forest-regressor")
            # instantiate the model and fit the model
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
    # define the search space
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }   
    # use the minimizing function of huper opt to minimize the objective function
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=25,
        help="the number of parameter evaluations for the optimizer to explore."
    )
    args = parser.parse_args()

    run(args.data_path, args.max_evals)





