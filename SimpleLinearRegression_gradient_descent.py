# prob : find the relationship between seniority &salary
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import matplotlib as mpl
import numpy as np

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)
# y = w * x + b
x = data["YearsExperience"]
y = data["Salary"]

def compute_gradient(x, y, w, b):
  # w_gradient = (2*x*(w*x+b-y)).mean() , 2 can be omitted because the constant will affect the step
  w_gradient = (x*(w*x+b-y)).mean()
  b_gradient = (w*x+b-y).mean()

  return w_gradient, b_gradient

def compute_cost(x, y, w, b):
  y_pred = w*x + b
  cost = (y - y_pred)**2
  cost = cost.sum() / len(x)
  return cost

def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter = 1000):
  c_hist = []
  w_hist = []
  b_hist = []

  w = w_init
  b = b_init
  learning_rate = 0.001

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
      print(f"Iteration {i:5} : Cost {cost:.4e}, w:{w:.2e}, b:{b:.2e}, w_gradient: {w_gradient:.2e}, b_gradient:{b_gradient:.2e}")

  return w, b, w_hist, b_hist, c_hist

w_init = 0
b_init = 0
learning_rate = 1.0e-3
run_iter = 20000
w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, learning_rate, compute_cost, compute_gradient, run_iter, p_iter = 1000)

print(f"final_w = {w_final:.2f}, final_b = {b_final:.2f}")
print(f"seniority = 3.5y salary:{w_final*3.5 + b_final} K ")
plt.plot(np.arange(0, 20000), c_hist)
plt.title("itertion vs cost")
plt.xlabel("itertion")
plt.ylabel("cost")
plt.show()

plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.view_init(20, -65)
ax.xaxis.set_pane_color((1, 1, 1))
ax.yaxis.set_pane_color((1, 1, 1))
ax.zaxis.set_pane_color((1, 1, 1))

ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

i = 0
for w in ws: 
  j = 0
  for b in bs: 
    cost = compute_cost(x, y, w, b)
    costs[i,j] = cost
    j = j+1
  i = i+1

b_grid, w_grid = np.meshgrid(bs, ws)
# https://wangyeming.github.io/2018/11/12/numpy-meshgrid/

ax.plot_surface(w_grid, b_grid, costs, alpha=0.3)

ax.set_title("w b -> cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")

w_index, b_index = np.where(costs == np.min(costs))
ax.scatter(ws[w_index], bs[b_index], costs[w_index, b_index], color="red", s=40)
ax.scatter(w_hist[0], b_hist[0], c_hist[0], color="green", s=40)
ax.plot(w_hist, b_hist, c_hist)

plt.show()
