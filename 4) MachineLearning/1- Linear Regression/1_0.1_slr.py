# single lr

from Functions.DataAnalysis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# dataset okuma
df = pd.read_csv("datasets/advertising.csv")
check_df(df)

# bağımlı ve bağımsız değişken
y = df[["sales"]]
X = df[["TV"]]


# görselleştirme
sns.jointplot(x="TV", y="sales",data=df, kind="reg")
plt.show()

# model y = wx + b
reg_model = LinearRegression().fit(X,y)
b = reg_model.intercept_[0] # b
w = reg_model.coef_[0][0] # w

# tahmin
y1 = w * 150 + b
y2 = w * 500 + b

reg_model.predict([[150]])
reg_model.predict([[500]])

new_df = [[150],[500]]
reg_model.predict(new_df)

new_df = pd.DataFrame(new_df)
reg_model.predict(new_df)

y_pred = reg_model.predict(X)

# model görselleştirme
g = sns.regplot(x=X, y=y, scatter_kws={'color':'b','s':9}, ci=False, color='r')
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


# tahmin başarısı
y_pred = reg_model.predict(X)
mse = mean_absolute_error(y,y_pred)
rmse = np.sqrt(mean_squared_error(y,y_pred))
rsquare = reg_model.score(X,y)
print(f"MSE: {mse}",
    f"RMSE: {rmse}",
    f"R-Square: {rsquare}")


