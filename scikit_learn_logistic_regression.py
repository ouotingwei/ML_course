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

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model.fit(x_train, y_train)
y_pred = lg_model.predict(x_test)
(y_pred==y_test).sum() / len(y_test)
