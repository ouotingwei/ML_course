import pandas as pd
url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv"
data = pd.read_csv(url)

# gendar boy = 1, girl = 0
data["Gender"] = data["Gender"].map({"男生" : 1, "女生" : 0})

from sklearn.model_selection import train_test_split
x = data[["Age", "Weight", "BloodSugar", "Gender"]]
y = data["Diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_train, x_test

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import numpy as np
w = np.array([1, 4, 2, 4])
b = 2

def sigmoid(z):
  return 1/(1+np.exp(-z))

def compute_cost(x, y, w, b):
  # sigmoid(z) # value -> 0~1
  #y_pred = sigmoid(z)
  # cost = -y*log(y_pred) - (1 - y)*log(1-y_pred)
  #cost = -y_train*np.log(y_pred) - (1-y_train)*np.log(1-y_pred)
  #cost.mean()
  z = (w*x).sum(axis=1) + b
  y_pred = sigmoid(z)
  cost = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
  return cost.mean()

a = compute_cost(x_train, y_train, w, b)
