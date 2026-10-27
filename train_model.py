import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([20, 30, 40, 50, 60, 70, 80, 90])

model = LinearRegression()
model.fit(X, y)

with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
