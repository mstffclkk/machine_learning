
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
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE (MSE değerini hesaplar) (güncellenen ağırlıkların hata oranına bakmak)
def cost_function(Y, b, w, X):
    """
    Cost function MSE

    Args:
        Y : bağımlı değişken
        b : sabit, bias
        w : weight, ağırlık
        X : bağımsız değişken

    Returns:
        mse: mean square error
    """
    m = len(Y)      # gözlem sayısı
    sse = 0         # sum of square error

    for i in range(m): #for i in range(1, m+1):
        y_hat = b + w * X[i]        # tahmin edilen değer
        y = Y[i]                    # gerçek değer
        sse += (y_hat - y) ** 2

    mse = sse / m   # ortalama hata
    return mse


# update_weights (ağırlıkları güncelleme)
def update_weights(Y, b, w, X, learning_rate):
    """
    update_weights

    Args:
        Y : bağımlı değişken
        b : sabit, bias
        w : weight, ağırlık
        X : bağımsız değişken
        learning_rate: öğrenme oranı

    Returns:
        new_b, new_w
    """
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """
    Train fonksiyonu

    Args:
        Y : bağımlı değişken
        initial_b (_type_): initial bias value
        initial_w (_type_): initial weight value
        X (_type_): bağımsız değişken
        learning_rate (_type_): öğrenme oranı
        num_iters (_type_): iterasyon sayısı

    Returns:
        cost_history, b, w
    """
    # ilk hatanın raporlandığı bölüm
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []       # hataları gözlemleyip saklamak için

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)

        # her 100 iterasyon da raporla 
        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    # iterasyon sayısı sonu raporlama
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

# normal denklemler yöntemiyle gradient descent arasında doğrusal regresyon açısından katsayı bulma ağırlık bulma açısından ne fark var?
# parametre: modelin veriyi kullanarak veriden hareketle bulduğu değerlerdir.(ağırlıklar w, b)
# hiperparametre: veri setinden bulunamayan, kullanıcı tarafından ayarlanması gereken değerlerdir.(initial_b, initial_w, X, learning_rate, num_iters)

df = pd.read_csv("datasets/advertising.csv")
check_df(df)

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 6000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

# hatanın düşmediği gözlemlenirse
# learning_rate değeri ile oynanabilir, yeni değişkenler eklenebilir

# görselleştirme
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Mean Square Error")
plt.title("Mean Square Error by Iteration")
plt.show()







