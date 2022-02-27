# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:11:20 2022

@author: Thobela Sixpence
"""


import numpy as np
import pickle 



# Loading the saved model
loaded_model = pickle.load(open('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Machine Learning Project 1 - Adult Salary Prediction/trained_model.sav', 'rb'))

# Test for a single individual using logistic regression
input_data = (51, 2,2,3,3,1,1,40.0)

# changing the data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)
if prediction == [1]:
    print("This individual qualifiers for premium prices: ", prediction)
else:
    print("This individual does not qualifiers for premium prices: ", prediction)
    
