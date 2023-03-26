import logging
import os
import pickle
import uuid

from flask import Flask, jsonify, request
from pymongo import MongoClient

# global references for variables
MONGO_ADDRESS = os.getenv("MONGO_ADDRESS", "mongodb://localhost:27017/")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "ride_prediction")
LOGGED_MODEL = os.getenv("MODEL_FILE", "lin_reg.bin")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# read the logged model
with open(LOGGED_MODEL,'rb') as f_in:
    dv, model = pickle.load(f_in)


# create a table in mongo DB
mongo_client = MongoClient(MONGO_ADDRESS)
# get the database 
mongo_db = mongo_client.get_database(MONGO_DATABASE)
# get the database collections
mongo_collection = mongo_db.get_collection('data')

# create a the flask web app
app = Flask("Ride-Prediction-Service")
logging.basicConfig(level=logging.INFO)

# prepare the input data and features that is being taken in
def prepare_features(ride):
    """Function to prepare features before making prediction"""
    # copy the input ride featues and data
    record = ride.copy()
    # from the list of features, create a table that has PO_DO feature comnbined
    record['PU_DO'] = '%s_%s' % (record['PULocationID'], record['DOLocationID'])
    # transform the featues
    features = dv.transform([record])

    return features, record

# after preparing featues they need to be sent to the database 
def save_db(record, pred_result):
    """Save data to mongo db collection"""
    # copy tthe transformed records
    rec = record.copy()
    # create a prediction column in the rec table and place the pred results
    rec["prediction"] = pred_result[0]
    # insert the data into the table
    mongo_collection.insert_one(rec)

@app.route("/", methods=["GET"])
def get_info():
    """Function to provide info about the app"""
    info = """<H1>Ride Prediction Service</H1>
              <div class="Data Request"> 
                <H3>Data Request Example</H3> 
                <div class="data">
                <p> "ride = {
                    "PULocationID": 10,
                    "DOLocationID": 50,
                    "trip_distance": 40
                    }"
                </p>
                </div>    
               </div>"""
    return info

@app.route("/predict-duration", methods=["POST"])
def predict_duration():
    """Function to predict duration"""
    # get the request from a jsson file
    ride = request.get_json()
    # prepare features
    features, record = prepare_features(ride)
    # predict using the model read above
    prediction = model.predict(features)
    # create a unique ride id
    ride_id = str(uuid.uuid4())
    # create a tuple of features for the predictions
    pred_data = {
            "ride_id": ride_id,
            "PU_DO": record["PU_DO"],
            "trip_distance": record["trip_distance"],
            "status": 200,
            "duration": prediction[0],
            "model_version": MODEL_VERSION
            }

    save_db(record, prediction)

    result = {
        "statusCode": 200,
        "data" : pred_data
        }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


