#############################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from Functions.DataAnalysis import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

#############################################
# Label Encoding & Binary Encoding(0-1 Encoding)
#############################################

df = load()
df.head()
df["Sex"].head()


le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5] # fit_transform: uygula ve dönüştür. Alfabetik sıraya göre 0 ve 1'e dönüştürdü.
le.inverse_transform([0, 1])     # inverse_transform: dönüştürülmüş hali geri dönüştür.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

df.head()

# nunique eksik değerleri saymaz. Eğer eksik değerler varsa, eksik değerleri dikkate almadan sadece unique değerleri sayar.
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())
