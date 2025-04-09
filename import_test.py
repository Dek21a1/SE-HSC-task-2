import rexport
import pickle
import numpy as np

rexport.save_model()

loaded_model = pickle.load(open('poly_model.sav', 'rb'))
predict = np.array([4]).reshape(-1, 1)
result = loaded_model.predict(predict)
print(result[0])