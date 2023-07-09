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

def compute_gradient(x, y, w, b):
  z = (w*x).sum(axis=1) + b
  y_pred = sigmoid(z)
  w_gradient = np.zeros(x.shape[1])
  b_gradient = (y_pred - y).mean()

  for i in range(x.shape[1]):
    w_gradient[i] = (x[:, i]*(y_pred - y)).mean()

  return w_gradient, b_gradient

def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter = 1000):
  c_hist = []
  w_hist = []
  b_hist = []

  w = w_init
  b = b_init

  for i in range(run_iter):
    w_gradient, b_gradient = gradient_function(x, y, w, b)
    w = w - w_gradient*learning_rate
    b = b - b_gradient*learning_rate
    cost = cost_function(x, y, w, b)

    w_hist.append(w)
    b_hist.append(b)
    c_hist.append(cost)
    #print("Ieration:", i, ", cost:", cost, ", w:", w, ", b:", b)
    if i%p_iter == 0:
      print(f"Iteration {i:5} : Cost {cost:2e}, w:{w}, b:{b}, w_gradient: {w_gradient}, b_gradient:{b_gradient}")

  return w, b, w_hist, b_hist, c_hist

w_init = np.array([1, 2, 2, 3])
b_init = 5
learning_rate = 1
run_iter = 10000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter, p_iter = 1000)

import matplotlib.pyplot as plt

# cost下降過程
plt.plot(np.arange(0, 100), c_hist[:100])
plt.title("iteration vs cost")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()

# w下降過程
plt.plot(np.arange(0, 100), w_hist[:100])
plt.title("iteration vs w")
plt.xlabel("iteration")
plt.ylabel("w")
plt.show()

# b下降過程
plt.plot(np.arange(0, 1000), b_hist[:1000])
plt.title("iteration vs b")
plt.xlabel("iteration")
plt.ylabel("b")
plt.show()

w_final, b_final

# test
z = (w_final*x_test).sum(axis=1) + b_final
y_pred = sigmoid(z)
y_pred = np.where(y_pred > 0.5, 1, 0) # y_pred>50% -> yes
accurate = (y_pred==y_test).sum() / len(y_test) * 100
print(f"correct rate {accurate}%")

# 72 92 102 girl
x_real = np.array([[72, 92, 102, 0]])
x_real = scaler.transform(x_real)
z = (w_final*x_real).sum(axis=1) + b_final
y_real = sigmoid(z)
print(y_real*100, "%")
