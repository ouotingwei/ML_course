# -*- coding: utf-8 -*-
"""pytorch_logistic_regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bnom2aIXSVq6W9I5QnaHs4qyyKbPVIM5
"""

import pandas as pd
url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Classification/Diabetes_Data.csv"
data = pd.read_csv(url)
data

# gendar boy = 1, girl = 0
data["Gender"] = data["Gender"].map({"男生" : 1, "女生" : 0})
data

from sklearn.model_selection import train_test_split
x = data[["Age", "Weight", "BloodSugar", "Gender"]]
y = data["Diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device

from torch import nn
class LogisticRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_layer = nn.Linear(in_features=4, out_features=1, dtype = torch.float64) # dtype error !
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.linear_layer(x)) # feature -> linear_layer -> sigmoid

torch.manual_seed(87)
model =  LogisticRegressionModel()
model = model.to(device)
model, model.state_dict()

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
x_train

x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
x_train

model(x_train)

y_train = y_train.reshape(-1, 1)
y_train = y_train.type(torch.double)  # change data type
y_test = y_test.reshape(-1, 1)
y_test = y_test.type(torch.double)  # change data type

# test

cost_fn = nn.BCELoss()
y_pred = model(x_train)
cost = cost_fn(y_pred, y_train)
# before
print(model.state_dict())
print(cost)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01) # SGD = gradient descent
optimizer.zero_grad() # gradient -> 0
cost.backward() # Calculate gradients
optimizer.step()  # update params

y_pred = model(x_train)
cost = cost_fn(y_pred, y_train)
# after
print(model.state_dict())
print(cost)

# cost : 0.7056 -> 0.7038
# w & b are different !
# its ok !

epochs = 10000    # Training iterations

train_cost_hist = []    # save training data
test_cost_hist = []
train_acc_hist = []     # Save prediction accuracy
test_acc_hist = []

for epoch in range(epochs):

  model.train()

  y_pred = model(x_train)   # predict

  train_cost = cost_fn(y_pred, y_train)   # compute cost
  train_cost_hist.append(train_cost.cpu().detach().numpy())   # cpu -> plot graphs only on the CPU.

  train_acc = (torch.round(y_pred)==y_train).sum() / len(y_train) * 100   # acc > 0.5 -> diabetes
  train_acc_hist.append(train_acc.cpu().detach().numpy())

  optimizer.zero_grad()     # Zeroing gradients

  train_cost.backward()     # Calculate gradients

  optimizer.step()          # update params

  model.eval()
  with torch.inference_mode():
    test_pred = model(x_test)                    # Predicting on the test set
    test_cost = cost_fn(test_pred, y_test)       # calculate costs
    test_cost_hist.append(test_cost.cpu())       # save costs

    test_acc = (torch.round(test_pred)==y_test).sum() / len(y_test) * 100
    test_acc_hist.append(test_acc.cpu())

  if epoch%1000==0:
    print(f"Epoch: {epoch:5}, Train Cost: {train_cost: .4e}, Test Cost: {test_cost: .4e}")
    print(f"Epoch: {epoch:5}, Train Acc: {train_acc}%, Test Acc: {test_acc}%")

# cost下降過程
import matplotlib.pyplot as plt
plt.plot(range(0, 10000), train_cost_hist, label="train cost")
plt.plot(range(0, 10000), test_cost_hist, label="test cost")
plt.title("train & test cost")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.plot(range(0, 10000), train_acc_hist, label="train acc")
plt.plot(range(0, 10000), test_acc_hist, label="test acc")
plt.title("train & test acc")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend()
plt.show()

# final param
model.state_dict()

model.eval()
with torch.inference_mode():
  y_pred = model(x_test)
(torch.round(y_pred)==y_test).sum() / len(y_test) * 100  # model accuracy
