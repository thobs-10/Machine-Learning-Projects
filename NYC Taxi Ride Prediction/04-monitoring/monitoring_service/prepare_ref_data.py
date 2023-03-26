import pandas as pd
import pyarrow.parquet as pq

# list the data that will be used and combined
data_files = ["../datasets/green_tripdata_2021-03.parquet", "../datasets/green_tripdata_2021-04.parquet"]
# the rsulting file format
output_file = "green_tripdata_2021-03to04.parquet"
# create a dataframe that will be used to concantenate the two datassets
df = pd.DataFrame()
# for each file read it to a pandas dataframe
for file in data_files:
    data = pq.read_table(file).to_pandas()
    # combine the two dataframaes
    df = pd.concat([data, df], ignore_index=True)
# save the combined  file
df.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)