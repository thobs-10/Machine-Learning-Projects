# import dependencies
import pandas as pd
import numpy as np
import pickle
import sys


# variables to be kept for ease of changes
# we only interested in 2021 data year = 2021
# for feb month = 2 

# the files to be inputed for the moddel to use and outputed version of the file
year = int(sys.argv[1]) # 2021
month = int(sys.argv[2]) #2
# tthe files are taken from the s3 bucket in AWS
input_file = f's3://nyc-tlc/trip data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'

# load the bin file that has both the model and the dict vectorizer using pickle
with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


print('predicted mean duration:', y_pred.mean())


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

