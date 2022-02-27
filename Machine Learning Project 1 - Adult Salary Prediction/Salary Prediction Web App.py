# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:22:27 2022

@author: Thobela Sixpence
"""

# import libraries
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd

# standard scaler to scale down the input values
sc = StandardScaler()

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 1 - Adult Salary Prediction/trained_model.sav', 'rb'))


# feature engineering
def feature_engineering(df):
    '''In this function we are going to be taking categorical features and turning them into numerical'''
    # convert the salary into 1 if salary is greater then 50K else 0
    #df['salary'] = df['salary'].replace(' >50K', '>50K')
    #df['salary'] = np.where(df['salary'] > '50K', 1, 0)
    
    # convert the sex column into 0 and 1, if male then 1 else 0
    # basically it says, in column 'sex', find a row where the value is 'Male', and replace that with 1. Replace 0 if other value
    # is there
    df['sex'] = np.where(df['sex'] == ' Male', 1, 0)
    # do the label encoding in race column (0: 'White',1: 'Black',2: 'Asian-Pac-Islander',3:'Amer-Indian-Eskimo',4:'Other')
    # the code says, for each unique value in column race, find the unique value and associate it with the unique key between(0, 1, 2,...)
    # and place that in label encoder race as a dictionary
    label_enco_race = {value: key for key, value in enumerate(df['race'].unique())}
    # now use that dictionary in column race to match the key with the values
    df['race'] = df['race'].map(label_enco_race)
    # {0: ' Not-in-family',1: ' Husband'2: ' Wife',3: ' Own-child',4: ' Unmarried',5: ' Other-relative
    label_enco_relation = {value: key for key, value in enumerate(df['relationship'].unique())}
    df['relationship'] = df['relationship'].map(label_enco_relation)
    
    # {0: ' Adm-clerical',1: ' Exec-managerial',2: ' Handlers-cleaners',3: ' Prof-specialty',4: ' Other-service',5: ' Sales', 6: ' Craft-repair',7: ' Transport-moving',8: ' Farming-fishing',9: ' Machine-op-inspct', 10: ' Tech-support', 11: ' ?',12: ' Protective-serv',13: ' Armed-Forces', 14: ' Priv-house-serv'}
    # in column occupation, where the value is '?', replce that with "missing", or if the value is not '?' just keep the value as it is
    df['occupation'] = np.where(df['occupation'] == ' ?', 'Missing', df['occupation'])
    label_enco_occu = {value: key for key, value in enumerate(df['occupation'].unique())}
    
    # Replacing ? value with 'Missing'
    df['occupation'] = df['occupation'].map(label_enco_occu)
    
    # {0: ' Never-married',1: ' Married-civ-spouse',2: ' Divorced',3: ' Married-spouse-absent',4: ' Separated',5: ' Married-AF-spouse',6: ' Widowed'}
    #label_enco_marital_status = {value: key for key, value in enumerate(df['marital_status'].unique())}
    #df['marital_status'] = df['marital_status'].map(label_enco_marital_status)
    
    #label_enco_edu = {value: key for key, value in enumerate(df['education'].unique())}
    #df['education'] = df['education'].map(label_enco_edu)
    
    # {0: ' State-gov'1: ' Self-emp-not-inc',2: ' Private',3: ' Federal-gov',4: ' Local-gov',5: ' ?',6: ' Self-emp-inc',7: ' Without-pay',8: ' Never-worked'}
    df['workclass'] = np.where(df['workclass'] == ' ?', 'Missing', df['workclass'])
    label_enco_workclass = {value: key for key, value in enumerate(df['workclass'].unique())}
    df['workclass'] = df['workclass'].map(label_enco_workclass)
    
    
    # {' United-States': 0,' Cuba': 1,' Jamaica': 2,' India': 3,' ?': 4,' Mexico': 5,' South': 6,' Puerto-Rico': 7,' Honduras': 8,' England': 9,' Canada': 10,' Germany': 11,' Iran': 12,' Philippines': 13,' Italy': 14,' Poland': 15,' Columbia': 16,' Cambodia': 17,' Thailand': 18,' Ecuador': 19,' Laos': 20,' Taiwan': 21,' Haiti': 22,' Portugal': 23,' Dominican-Republic': 24,' El-Salvador': 25,' France': 26,' Guatemala': 27,' China': 28,' Japan': 29,' Yugoslavia': 30,' Peru': 31,' Outlying-US(Guam-USVI-etc)': 32,' Scotland': 33,' Trinadad&Tobago': 34,' Greece': 35,' Nicaragua': 36,' Vietnam': 37,' Hong': 38,' Ireland': 39,' Hungary': 40,' Holand-Netherlands': 41
    #df['native_country'] = np.where(df['native_country'] == ' ?', 'Missing', df['native_country'])
    #label_enco_workclass = {value: key for key, value in enumerate(df['native_country'].unique())}
    #df['native_country'] = df['native_country'].map(label_enco_workclass)
    
    return df

   
# create function for prediction
def premium_prediction(input_data):
    
    # changing the data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    print(prediction)
    if prediction == [1]:
        return "This individual qualifiers for premium prices." 
    else:
        return "This individual does not qualifiers for premium prices."


# using the streamlit library

def main():
    
    # title for wep app
    st.title('Premium Salary Classification Wep App')
    
    # getting the input data from the user
    
    
    Age = st.text_input('Age:')
    workclass = st.text_input('workclass:')
    #fnlwgt = st.text_input('fnlwgt value:')
    #education = st.text_input('education:')
    education_num = st.text_input('education number:')
    #marital_status = st.text_input('marital Status:')
    occupation = st.text_input('occupation:')
    relationship = st.text_input('relationship:')
    race = st.text_input('race:')
    sex = st.text_input('sex')
    #capital_gain = st.text_input('capital gain')
    #capital_loss = st.text_input('capital loss')
    hours_per_week = st.text_input('hours per week')
    #native_country = st.text_input('native_country')
    
    
    
    # code for prediction
    result = ""
    
    
    # feature engineering
    temp_input = [Age, workclass, education_num,  occupation,
                  relationship, race, sex,  hours_per_week]
    
    temp_df = pd.DataFrame(temp_input).T
    #fix columns 
    temp_df.columns = ['age','workclass', 'education_num', 'occupation',
             'relationship', 'race', 'sex', 'hours_per_week']
    # place in the feature engineering function
    final_input = feature_engineering(temp_df)
    final_input_list = final_input.iloc[:,:].values
    final_input_list = final_input_list.flatten()
    # tuple input for input data
    tuple_input_data = tuple(final_input_list)
    
    # final prediction
    if st.button('Premium Test Result'):
        result = premium_prediction(tuple_input_data)
        
    
    st.success(result)
    
    

if __name__ == '__main__':
    main()
    
    
    
    
    