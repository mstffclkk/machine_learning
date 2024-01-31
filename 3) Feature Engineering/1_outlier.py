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

from Functions.DataAnalysis import * # Functions/DataAnalysis.py Imports all functions in the file.

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

a, b, c = grab_col_name(df)

list(map(lambda x: cat_summary(df, x, plot=False), cat_cols))
list(map(lambda x: num_summary(df, x, plot=False), num_cols))

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    print(col, outlier_thresholds(df, col))

for col in num_cols:
    print(col, "-->", grab_outliers(df, col, True))  




