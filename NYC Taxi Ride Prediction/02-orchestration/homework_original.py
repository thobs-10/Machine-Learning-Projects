# import libraries
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# reading the data paths from the folder and paquet files
def read_data(path):
    df = pd.read_parquet(path)
    return df


# data preprocessing for train and validation datasets
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


# training the chosen model, in this case the linear reg model
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv


# this is the valdation stage of the model
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

# this main takes in string paths to our data into the variables to be used by read_data function
def main(train_path: str = './data/fhv_tripdata_2021-01.parquet', 
           val_path: str = './data/fhv_tripdata_2021-02.parquet'):

    # declare the features of interest
    categorical = ['PUlocationID', 'DOlocationID']
    # read the data fo train dataset
    df_train = read_data(train_path)
    # preprocess the train dataset
    df_train_processed = prepare_features(df_train, categorical)
    # get the ddata for validation
    df_val = read_data(val_path)
    # preprocess it by using  transform for dict vect not the fit_transform since we know that it is false and not train data
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    # perform validation of model
    run_model(df_val_processed, categorical, dv, lr)

main()





