import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)
# y = w * x + b
x = data["YearsExperience"]
y = data["Salary"]

def compute_cost(x, y, w, b):
  y_pred = w*x + b
  cost = (y - y_pred)**2
  cost = cost.sum() / len(x)
  return cost

costs = []
for w in range(-100, 101):
  cost = compute_cost(x, y, w, 0)
  costs.append(cost)

#plt.scatter(range(-100, 101), costs)
plt.plot(range(-100, 101), costs)
plt.title("cost function b=0 w=-100~100")
plt.xlabel("w")
plt.ylabel("cost")
plt.show()

ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

i = 0
for w in ws:
  j = 0
  for b in bs:
    cost = compute_cost(x, y, w, b)
    costs[i][j] = cost
    j = j+1
  
  i = i+1

ax = plt.axes(projection="3d")
ax.xaxis.set_pane_color((1, 1, 1))
ax.yaxis.set_pane_color((1, 1, 1))
ax.zaxis.set_pane_color((1, 1, 1))

w_grid, b_grid = np.meshgrid(bs, ws) #2d網格
ax.plot_surface(w_grid, b_grid, costs, cmap = "Spectral_r", alpha = 0.7)
ax.plot_wireframe(w_grid, b_grid, costs, color="black", alpha = 0.3)

plt.figure(figsize = (10, 10))
ax.view_init(45, -120)
ax.set_title("w b -> cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")

w_index, b_index = np.where(costs == np.min(costs))
print(np.min(costs))
print("when w = ", ws[w_index], ", b = ", bs[b_index], "the cost is the smallest")
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color = "red", s = 40)

plt.show()
