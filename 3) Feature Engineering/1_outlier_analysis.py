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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv") 
    return data

df = load()
df.head()

sns.boxplot(x=df["Age"])
plt.show()

#############################################
# Finding Outliers
#############################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up_limit = q3 + 1.5 * iqr
low_limit = q1 - 1.5 * iqr

df[(df["Age"] < low_limit) | (df["Age"] > up_limit)] # outlier values
df[(df["Age"] < low_limit) | (df["Age"] > up_limit)].index # outlier index
df[(df["Age"] < low_limit) | (df["Age"] > up_limit)].any(axis=None) # outlier var mı yok mu

# Function

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

#grab_col_names ten sonra num_cols ile kullanıcaz
#for col in num_cols:
#    print(col, outlier_thresholds(df, col)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

#grab_col_names ten sonra num_cols ile kullanıcaz
#for col in num_cols:
#    print(col, check_outlier(df, col)

#############################################
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}\nVariables: {dataframe.shape[1]}\ncat_cols: {len(cat_cols)}\n"
          f"num_cols: {len(num_cols)}\ncat_but_car: {len(cat_but_car)}\nnum_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# num_cols PassengerId yi yakalamış. Bunun için num_cols listesinden çıkartıyoruz.
num_cols = [col for col in num_cols if col not in "PassengerId"]

# nümerik değişkenler için check_outlier fonksiyonunu çağırıyoruz./
for col in num_cols:
    print(col, check_outlier(df, col))
#############################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
age_index = grab_outliers(df, "Age", True)

###############################################################
### SİLME VEYA BASKILAMA YÖNTEMLERİNDEN BİRİ KULLANILIR.
###############################################################
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]
###############################################################

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df.shape

# veri setinde outlier check edilir.( age True , fare True)
for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değişkenleri threshold ile değiştir.
for col in num_cols:
    replace_with_thresholds(df, col)

# aykırı değer var mı kontrol et.
for col in num_cols:
    print(col, check_outlier(df, col))

#output false, false


###################
# Recap
###################

df = load()

# saptama işlemleri
outlier_thresholds(df, "Age")           # aykırı değeri saptamak ve bu saptama işlemi için gerekli thresholdları bulmak.
check_outlier(df, "Age")                # bu thresholdlara göre outlier var mı yok mu kontrol et.
grab_outliers(df, "Age", index=True)    # outlierları getir.


# aykırı değerleri silme veya baskılama yöntemi
remove_outlier(df, "Age").shape          # aykırı değerleri silme ama atama yapmadık
replace_with_thresholds(df, "Age")       # aykırı değerleri baskılama, threshold ile değiştir.
check_outlier(df, "Age")                 # aykırı değerleri kontrol et.

a = [2,4,6,8]
 
for i in a:
    print(i**2)