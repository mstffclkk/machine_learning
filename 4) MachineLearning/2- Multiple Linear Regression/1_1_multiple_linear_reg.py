from Functions.DataAnalysis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# Multiple Linear Regression

######################################################
df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)        # bağımlı değişkeni atıp kaydedersek bağımsız değişkenlerin hepsini seçmiş oluruz.

y = df[["sales"]]                   # bağımlı değişken

##########################
# Model
##########################
# statsmodels ile model kurma
import statsmodels.api as sm
model = sm.OLS(y, X).fit()

model.summary()
#############################

# train_test_split: train ve test setine ayırmayı sağlar.Bağımlı ve bağımsız değişkenleri alır.
# test_size=0.20: test setinin boyutunu %20 yapar.(train seti %80 olmuş olur)
# train_test_split bu fonksiyonun çıktısı.
# train setinde X, y test setinde X, y verir.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape 
y_train.shape

# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias - intercept)
reg_model.intercept_        # 2.90

# coefficients (w - weights - coef)
reg_model.coef_             # tv: 0.0468431 , radio: 0.17854434, newspaper: 0.00258619


##########################
# Tahmin
##########################
# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# b: 2.90
# coef : tv: 0.0468431 , radio: 0.17854434, newspaper: 0.00258619
# model denklemi --> Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619 # tahmin edilen satış

# fonksiyonel olarak yazarsak
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri) # 6.202131

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
mean_squared_error(y_train, y_pred)    

# Train RKARE
reg_model.score(X_train, y_train)               # 0.8959372632325174

# Test RMSE 
y_pred = reg_model.predict(X_test)
mean_squared_error(y_test, y_pred, squared=False)   # 1.41

# normalde test rmse > train rmse çıkması lazım ama düşük çıkmış güzel bir durum

# Test RKARE
reg_model.score(X_test, y_test)                 # 0.8927605914615384


# cross validation
# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X, # X_train 
                                 y, # y_train
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69



# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


