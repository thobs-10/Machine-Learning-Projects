import json
import uuid
from datetime import datetime

import pyarrow.parquet as pq
import requests

# read the parquet file and convert it into pandas also sampling only 5000 data points or rows
table = pq.read_table("../datasets/green_tripdata_2021-05.parquet")\
          .to_pandas()\
          .sample(n=5000, random_state=42) #5000 rows sampled
# make a copy of the table data
data = table.copy()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

# open a targer csv file, traverse it.
with open("target.csv", 'w') as f_target:
    for index, row in data.iterrows():
        # create a unique row identifier
        row['id'] = str(uuid.uuid4())
        # for each row, transfrom the required features, annd making the new duration column to be in minutes
        duration = (row['lpep_dropoff_datetime'] - row['lpep_pickup_datetime']).total_seconds() / 60
        # make sure the mminutes are greater than 1 or less than 60 minutes
        if duration >= 1 and duration <= 60:
            # write the results to the file
            f_target.write(f"{row['id']},{duration}\n")
        # post the results in a request post
        resp = requests.post("http://127.0.0.1:9696/predict-duration",
                             headers={"Content-Type": "application/json"},
                             data=row.to_json()).json()
        print(f"prediction: {resp['data']['duration']}")
