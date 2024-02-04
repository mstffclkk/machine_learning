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

df = load()
df.head()
check_df(df)


#############################################
# Advanced Analytics
#############################################

###################
# Examining the Missing Data Structure
###################

msno.bar(df) # veri setindeki tam gözlem sayısını gösterir.
plt.show()

msno.matrix(df) # downgrade matplotlib to 3.6 or upgrade missingno to 0.5.2
plt.show()

msno.heatmap(df)
plt.show()

###################
# Examining the Relationship of Missing Values with Dependent Variable
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)

df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

missing_vs_target(df, "Survived", na_cols)


