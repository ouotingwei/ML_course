import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data2.csv"
data = pd.read_csv(url)

#label encoding
data["EducationLevel"] = data["EducationLevel"].map({"高中以下":0, "大學":1, "碩士以上":2})
data

# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder()
onehot_encoder.fit(data[["City"]]) # fit -> 2d_array
city_encoded = onehot_encoder.transform(data[["City"]]).toarray()

data[["CityA", "CityB", "CityC"]] = city_encoded
data = data.drop(["City", "CityC"], axis=1) # axis = 1 -> raw

x = data[["YearsExperience", "EducationLevel", "CityA", "CityB"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def compute_cost(x, y, w, b):
  y_pred = (x * w).sum(axis = 1) + b
  cost = ((y - y_pred)**2).mean()

  return cost

# w1_gradient = x1*(y_pred - y_train)
def compute_gradient(x, y, w, b):
  y_pred = (x*w).sum(axis=1) + b
  w_gradient = np.zeros(x.shape[1])

  for i in range(x.shape[1]):
    w_gradient[i] = (x[:, i]*(y_pred - y)).mean()

  b_gradient = (y_pred - y).mean()

  return w_gradient, b_gradient

# w = np.array([1, 2, 2, 4])
# b = 1
# learning_rate = 0.001
# w_gradient, b_gradient = compute_gradient(x_train, y_train, w, b)
# print(compute_cost(x_train, y_train, w, b))
# w = w - w_gradient*learning_rate
# b = b - b_gradient*learning_rate
# w, b
# print(compute_cost(x_train, y_train, w, b))


np.set_printoptions(formatter={'float': '{: .2e}'.format})
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

w_init = np.array([1, 2, 2, 4])
b_init = 0
learning_rate = 1.0e-2
run_iter = 10000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x_train, y_train, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter, p_iter = 1000)

compute_cost(x_test, y_test, w_final, b_final) # cost:18 < 25

#test
# YearsExperience:5.3 edu:master locate:cityA
# YearsExperience:7.2 edu:high school locate:cityB
x_real = np.array([[5.3, 2, 1, 0], [7.2, 0, 0, 1]])
x_real = scaler.transform(x_real)
y_real = (w_final*x_real).sum(axis=1) + b_final
y_real
