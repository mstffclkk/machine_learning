#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)

#############################################
# 1. Genel Resim
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()                   # ilk 5 gözlem
df.tail()                   # son 5 gözlem
df.shape                    # boyut bilgisi
df.info()                   # veri seti hakkında bilgi
df.columns                  # değişken isimleri
df.index                    # index bilgisi
df.describe().T             # betimsel istatistikler
df.isnull().values.any()    # herhangi bir eksik gözlem var mı?
df.isnull().sum()           # her bir değişkende kaçar tane eksik gözlem var?
df.isnull().sum()[df.isnull().sum() != 0] # her bir değişkende (0 olmayanlar hariç) eksik gözlemleri göster.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("\n##################### Info #####################")
    print(dataframe.info())
    print("\n##################### Types #####################")
    print(dataframe.dtypes)
    print("\n##################### Head #####################")
    print(dataframe.head(head))
    print("\n##################### Tail #####################")
    print(dataframe.tail(head))
    print("\n################ Null Values ##################")
    print(dataframe.isnull().values.any())
    print("\n##################### NA #####################")
    print(dataframe.isnull().sum())
    print("\n##################### Not in 0 for NA #####################")
    print(dataframe.isnull().sum()[dataframe.isnull().sum() != 0])
    print("\n##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("\n##################### Value Counts #####################")
    print([dataframe[col].value_counts() for col in dataframe.columns if dataframe[col].nunique() < 10])



check_df(df)

