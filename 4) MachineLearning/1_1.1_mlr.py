from Functions.DataAnalysis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")
X = df.drop("sales", axis=1)
y = df[["sales"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=1
)

reg_model = LinearRegression().fit(X_train, y_train)

y_pred_test = reg_model.predict(X_test)
y_pred_train = reg_model.predict(X_train)

rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)


def print_metrics(rmse_test, rmse_train):
    print(f"RMSE for test set: {rmse_test}")
    print(f"RMSE for train set: {rmse_train}")


print_metrics(rmse_test, rmse_train)

# Cross-validation
cv_mean = np.mean(
    np.sqrt(
        -cross_val_score(
            reg_model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"
        )
    )
)
print(f"Cross-validation RMSE: {cv_mean}")


