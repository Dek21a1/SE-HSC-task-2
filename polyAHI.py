# Import frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

training_data = pd.read_csv('model_development/model_ready_data.csv')

x_name = ['Risk%']
y_name = 'AHI'
x = np.array(training_data[x_name])
y = np.array(training_data[y_name])


poly = PolynomialFeatures(degree=7, include_bias=False)
test_poly = PolynomialFeatures(degree=7, include_bias=False)
poly_features = poly.fit_transform(x)

# Create the model
my_model = LinearRegression()
# Fit the model to the data
my_model.fit(poly_features, y)


Age = input("Enter your age: ")
BMI = input("Enter your BMI: ")
# Convert inputs to float
Age = float(Age)
BMI = float(BMI)

#the minimum value with space for outliers
MIN_BMI = 17
#the maximum value with space for outliers
MAX_BMI = 48
#scale features
BMI = (BMI - MIN_BMI) / (MAX_BMI - MIN_BMI)

data_frame = pd.read_csv('model_development/model_ready_data.csv')
Risk = BMI * Age
input_value = (Risk / data_frame['Risk'].max())



# Define a function to predict based on input
def predict_value(input_value):
    # Ensure the input is in the correct format
    input_array = np.array(input_value).reshape(-1, 1)
    # Transform the input using the polynomial features
    input_poly = poly.transform(input_array)
    # Predict using the trained model
    prediction = my_model.predict(input_poly)
    return prediction


# Example: Predict for a given Risk% value
input_value = [[input_value]]  # Replace 50 with your desired input
predicted_value = predict_value(input_value)

#the minimum value with space for outliers
MIN_AHI = 0
#the maximum value with space for outliers
MAX_AHI = 100
#scale features
AHI = (predicted_value[0] + MIN_AHI) * (MAX_AHI + MIN_AHI)

print(f"Predicted AHI for Risk% {input_value[0][0]}: {AHI}")