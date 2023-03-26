import pandas as pd
import numpy as np
import pickle
import os 
import argparse
from sklearn.feature_extraction import DictVectorizer

# save the dataset to the data folder
def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

# read the dataset from the data folder and preprocess it
def read_dataframe(filename: str):
    # the file is in parquet so read it in parquet
    df = pd.read_parquet(filename)
    # get the duration colummn thaat indicates how long a particular drivee took
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    # since the drives some of them are in minutes some which are few are in hours. lets filteer
    # and get those trips that are in 
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    # get duratio that are greater than 1 minute and less tthan an hour
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    # get the unique id of pick up and drop off and make them strings fo they can be categorical
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    # combine the drop off and pick up location ID's
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    # sset the new column as categorical column
    categorical = ['PU_DO']
    # set the new trip distance as the numerical distance
    numerical = ['trip_distance']
    # create a dict that has PU_DO and trip distance as values
    dicts = df[categorical + numerical].to_dict(orient='records')
    # convert the dict into a vectorizer of a sparse matrix
    if fit_dv:
        # if yes, then that is a training data and needs to be fitted
        X = dv.fit_transform(dicts)
    else:
        # the data is for testing and needs to be transformed only
        X = dv.transform(dicts)
    return X, dv


def run(raw_data_path: str, dest_path: str, dataset: str = "green"):
    # load parquet files for training
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2021-01.parquet")
    )
    # load parquet files for validation
    df_valid = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2021-02.parquet")
    )
    # load parquet files for testing
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2021-03.parquet")
    )

    # extract the target from all the dataframes
    target = 'duration'
    y_train = df_train[target].values
    y_valid = df_valid[target].values
    y_test = df_test[target].values

    # fit the dictvectorizer and preprocess data
    dv = DictVectorizer()
    # set true for the fit_dv data for training and false for testing and validation
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_valid, _ = preprocess(df_valid, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save dictvectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="the location where the raw NYC taxi trip data was saved"
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)


