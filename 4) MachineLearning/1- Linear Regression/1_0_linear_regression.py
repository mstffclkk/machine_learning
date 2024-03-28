######################################################
# Sales Prediction with Linear Regression
######################################################
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
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
df = pd.read_csv("datasets/advertising.csv")
check_df(df)

X = df[["TV"]]              # bağımsız değişken
y = df[["sales"]]           # bağımlı değişken

# bağımsız değişkenin bağımlı değişken üzerindeki etkisini görmek için görselleştirme.
sns.jointplot(x="TV", y="sales", data=df, kind="reg")
plt.show()


##########################
# Model
##########################
""" 

y_hat = b + w*X (tek değişkenli linear reg model formulasyonu)
b: sabit (b - bias)
w: ağirlik (teta, w, coefficient)
x: bağimsiz değişken
""" 

# reg = LinearRegression()        # modeli oluşturduk
# model = reg.fit(X, y)           # modeli fit ettik

# model kurma ve fit etme
reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
reg_model.intercept_[0]             # array olduğu için 0. elemanı seçiyoruz.

# (x) in tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################
# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?
reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T


# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
# ci: confidence interval(güven aralığı)

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

   

##########################
# Tahmin Başarısı
##########################
"""
mean_squared_error(y, y_pred)
y: gerçek değer
y_pred: tahmin edilen değer  (reg_model.predict(X))

reg_model.predict(X) --> burada X yukarıda atamış olduğumuz TV sütunu. Diyoruz ki sen bu tv derğerlerini al bana tahmini
                         sales değerlerini ver.( yani bağımsız değikeni al, elimizde yokmuş gibi bağımlı değişkenleri tahmin et)

"""

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)           # 10.51

y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))  # 3.24

# MAE
mean_absolute_error(y, y_pred)          # 2.54

# R-KARE ( )
reg_model.score(X, y)                   # 0.612

# Not: değişken sayısı arttıkça r-kare şişmeye meyillidir. düzeltilmiş r-kare değerinin de göz önünde bulundurulması gerekir.
# Not: istatistiki çıktılarla ilgilenmiyoruz.
# r-kare değeri 1'e yaklaştıkça modelin başarısı artar.

