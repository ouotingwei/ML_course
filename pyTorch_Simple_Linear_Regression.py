import torch
import pandas as pd
from torch import nn # Neural networks
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)
# y = w * x + b
x = data["YearsExperience"]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=87)

# data -> numpy 
x_train = x_train.to_numpy()  
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# numpy -> pytorch
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

class LinearRegressionModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.rand(1, requires_grad=True))
    self.b = nn.Parameter(torch.rand(1, requires_grad=True))

  def forward(self, x):
    return self.w * x + self.b

torch.manual_seed(87)
model = LinearRegressionModel()
#list(model.parameters())
#model.state_dict()

# compute loss
cost_fn = nn.MSELoss()
"""
y_pred = model(x_train)
cost = cost_fn(y_pred, y_train)
print(model.state_dict())
print(cost)
"""

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03)
"""
optimizer.zero_grad()
# compute gradient
cost.backward() 
# update parameters  
optimizer.step()

y_pred = model(x_train)
cost = cost_fn(y_pred, y_train)
print(model.state_dict())
print(cost)
"""

epochs = 1000
train_cost_hist = []
test_cost_hist = []

for epoch in range(epochs):

  model.train() # mark : training

  y_pred = model(x_train)

  train_cost = cost_fn(y_pred, y_train)
  train_cost_hist.append(train_cost.detach().numpy())

  optimizer.zero_grad()
  train_cost.backward() 

  optimizer.step()

  model.eval()  # mark : testing
  # speeeeeed up !
  with torch.inference_mode():

    test_pred = model(x_test)

    test_cost = cost_fn(test_pred, y_test)
    test_cost_hist.append(test_cost.detach().numpy())

  if epoch%100 == 0:
    print(f"Epoch: {epoch:5d} Train_cost: {train_cost.item():.4e}, Test_cost: {test_cost:.4e}")

# show the figure
plt.plot(range(0, 1000), train_cost_hist, label="train cost")
plt.plot(range(0, 1000), test_cost_hist, label="test cost")
plt.title("train & test_cost")
plt.xlabel("epochs")
plt.ylabel("cost")
plt.legend()
plt.show()

model.state_dict()
