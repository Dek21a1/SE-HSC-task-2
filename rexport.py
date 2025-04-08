import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle


def save_model():
    training_data = pd.read_csv('training_data.csv', delimiter=',')
    x = np.array(training_data[:,1]).reshape(-1, 1)
    y = np.array(training_data[:,0])

    my_model = LinearRegression()
    my_model.fit(x, y)

    filename = 'poly_model.sav'
    pickle.dump(my_model, open(filename, 'wb'))