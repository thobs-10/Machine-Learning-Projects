# import dependencies
import json
import os
import pickle
import pandas as pd
from datetime import datetime

import pandas
import pyarrow.parquet as pq
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, RegressionPerformanceMetrics
from evidently.metrics import RegressionErrorPlot, RegressionErrorDistribution
from evidently.metric_preset import TargetDriftPreset
from evidently.report import Report


from prefect import flow, task

from pymongo import MongoClient


MONGO_CLIENT_ADDRESS = "mongodb://localhost:27017/"
MONGO_DATABASE = "prediction_service"
PREDICTION_COLLECTION = "data"
REPORT_COLLECTION = "report"
REFERENCE_DATA_FILE = "../data/green_tripdata_2021-03.parquet"
TARGET_DATA_FILE = "target.csv"
MODEL_FILE = os.getenv('MODEL_FILE', '../prediction_service/lin_reg.bin') 

# create a task for prefect, a function to upload the target variable
#@task
def upload_target(filename):
    # declare the mongo client by accessing the mongo client address
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    # get  the dataase and assigining the collection a collection table name
    collection = client.get_database(MONGO_DATABASE).get_collection(PREDICTION_COLLECTION)
    # open the uploaded file
    with open(filename) as f_target:
        # read each line, for each line
        for line in f_target.readlines():
            # split the row since it is a csv file
            row = line.split(",")
            # access the id and the target, place them in the collection table called data
            collection.update_one({"id": row[0]},
                                  {"$set": {"target": float(row[1])}}
                                 )
# task to load reference data
#@task
def load_reference_data(filename):
    # load the trained and tested model, with its pickle file
    with open(MODEL_FILE, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    # read the ref data from the filename, coinvert it to pandas and sample 5000 rows
    reference_data = pq.read_table(filename).to_pandas().sample(n=5000,random_state=42) #Monitoring for 1st 5000 records
    # Create features
    reference_data['PU_DO'] = reference_data['PULocationID'].astype(str) + "_" + reference_data['DOLocationID'].astype(str)

    # add target column which is the duration
    # reference_data['lpep_dropoff_datetime'] = reference_data['lpep_dropoff_datetime'].astype(datetime)
    # reference_data['lpep_pickup_datetime'] = reference_data['lpep_pickup_datetime'].astype(datetime)
    reference_data['target'] = reference_data.lpep_dropoff_datetime - reference_data.lpep_pickup_datetime
    # transform the target feature into minutes
    reference_data.target = reference_data.target.apply(lambda td: td.total_seconds() / 60)
    # make sure the minutes are greater than 1 and less than 60
    reference_data = reference_data[(reference_data.target >= 1) & (reference_data.target <= 60)]
    # create a list of teh features that will be used or are of interest
    features = ['PU_DO', 'PULocationID', 'DOLocationID', 'trip_distance']
    # transform the features usingg the dict vectorizer
    x_pred = dv.transform(reference_data[features].to_dict(orient='records'))
    # place the rpediction in a columnn of the reference data
    reference_data['prediction'] = model.predict(x_pred)
    return reference_data

# function to fetch the data
#@task
def fetch_data():
    '''create a dataframe of the data that i stored in mongo db'''
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    data = client.get_database(MONGO_DATABASE).get_collection(PREDICTION_COLLECTION).find()
    df = pandas.DataFrame(list(data))
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df.drop(['_id'], axis = 1, inplace=True)
    return df


#@task
def run_evidently(ref_data, data):
    # from both the ref data and original data drop the ehail fee feature
    ref_data.drop(['ehail_fee'], axis=1, inplace=True)
    data.drop('ehail_fee', axis=1, inplace=True)  # drop empty column (until Evidently will work with it properly)

    # monitor data drift
    data_drift_report = Report(metrics=[DataDriftTable(),TargetDriftPreset()])
    data_drift_report.run(reference_data=ref_data, current_data=data)
    # monitor model drift
    data_drift_report.save_html(filename='car_price_data_drift.html')

    # create a profile for the ride prediction data
    #profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    # create a mapping for thrr features that are main focal point
    #  mapping = ColumnMapping(prediction="prediction", numerical_features=['trip_distance'],
    #                         categorical_features=['PULocationID', 'DOLocationID'],
    #                         datetime_features=[])
    # map the ref data, data and the mapping of features to calcutae the necessary  metrics
    #profile.calculate(ref_data, data, mapping)
    # declare a dashboard, a regression performance tab
    #dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    # calculate the metrics in the dashboard
    #dashboard.calculate(ref_data, data, mapping)
    # return the profile json and the dashboard
    return json.loads(data_drift_report.json())


#@task
def save_report(result):
    """Save evidendtly profile for ride prediction to mongo server"""
    client = MongoClient(MONGO_CLIENT_ADDRESS)
    collection = client.get_database(MONGO_DATABASE).get_collection(REPORT_COLLECTION)
    collection.insert_one(result)

#@task
def save_html_report(result, filename_suffix=None):
    """Create evidently html report file for ride prediction"""
    
    if filename_suffix is None:
        filename_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    #result.save_html(f"ride_prediction_drift_report_{filename_suffix}.html")


#@flow
def batch_analyze(filename_suffix=None):
    # upload the  taget csv file 
    upload_target(TARGET_DATA_FILE)
    # get the ref data
    ref_data = load_reference_data(REFERENCE_DATA_FILE)
    # fetch the data
    data = fetch_data()
    # getr teh evidently profile and dashboard
    profile = run_evidently(ref_data, data)
    # save the batch monitoring dashboard results html
    # if filename_suffix is None:
    #     filename_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M')
    #     dashboard.save_html("ride_prediction_drift_report.html")
    # save the report
    save_report(profile)
    #save_html_report(dashboard)

batch_analyze()


