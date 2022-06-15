
from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result",methods=['POST','GET'])
def result():
    avg_glucose_level = float(request.form['avg_glucose_level'])
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    gender = int(request.form['gender'])

    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])




    x_values = np.array([avg_glucose_level, age, bmi, smoking_status, work_type, Residence_type,
                         gender]).reshape(1, -1)
    x = [[avg_glucose_level, age, bmi, smoking_status, work_type, Residence_type, gender]]

    sc = None

    scaler_path = os.path.join('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 8 - Stroke Prediction/models/',
                             'scaler.pkl')
    scaler=None
    with open(scaler_path, 'rb') as scaler_file:
        sc = pickle.load(scaler_file)

    x = sc.transform(x_values)

    model_path = os.path.join('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 8 - Stroke Prediction/models/',
                            'stroke_model.plk')
    model = joblib.load(model_path)

    y_pred = model.predict(x)

    # for No Stroke Risk
    if y_pred == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')


if __name__ == "__main__":
    app.run(debug=True, port=7384)


