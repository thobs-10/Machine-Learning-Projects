import pickle

import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# it takes ina file of which is in paquet format, transfroms the data and returns a required df
def read_dataframe(filename):
    # reading the paquet file to a pandas df
    df = pq.read_table(filename).to_pandas()
    #changeing the datatypes of the pick and drop off features to datetime
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    # creating a target variable from the drop of and the pick up time Which is the duration between
    # the pick up and the drop off time
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    # since the data is in hours convert it into minutes
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    # only focus on the duration that took more than a minute and less than an hour
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    # get the categorical features that wil make up the required dataframe
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

# ge the data or the additional data transforms it and outputs x_train , y_train and the dict_victorizer that was used
def add_features(train_data="./data/green_tripdata_2021-02.parquet",
                 additional_training_data=None):
    # reads the paquet file using the above function
    df_train = read_dataframe(train_data)
    # if the additional data has been passed, if yes then read and combine it with the original passed data
    if additional_training_data:
        extra_data = read_dataframe(additional_training_data)
        df_train = pd.concat([df_train, extra_data], axis=0, ignore_index=True)


    # formatting the data to have a column combining both the drop off and pick up llocation
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    # make the combined feature column the categorical since it is in datetime and in string datatype
    categorical = ['PU_DO'] 
    # get the numerical feature 
    numerical = ['trip_distance']

    dv = DictVectorizer()
    #conver the dataframe to a dictionary of features with their values since the dict vectorizer only takes dict
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    # apply the dict vectorizer 
    X_train = dv.fit_transform(train_dicts)
    # create ay -train numpy array 
    target = 'duration'
    y_train = df_train[target].values

    return X_train, y_train, dv




if __name__ == "__main__":
    # get the features from the original paquet file
    X_train, y_train, dv = add_features()
    
    print("Training model with one month of data")
    lr = LinearRegression()
    # model v1
    lr.fit(X_train, y_train)

    # saving the model and the dict vectorizer
    with open('prediction_service/lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)
    # get the features from the original paquet file and the additional data
    X_train, y_train, dv = add_features(additional_training_data="./data/green_tripdata_2021-03.parquet")
    print("Training model with two months of data")
    lr = LinearRegression()
    # modelv2
    lr.fit(X_train, y_train)
     # saving the model and the dict vectorizer
    with open('prediction_service/lin_reg_V2.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)
