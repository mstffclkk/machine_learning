import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

def check_df(dataframe, head=5):
    print("\n##################### Shape #####################")
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
    print("\n##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



def grab_col_names(dataframe, cat_th=10,  car_th=20):
    
    # cat_cols, cat_but_car: Analysis of Categorical Variables
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th 
                   and dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th 
                   and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols Analysis of Numerical Variables
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    num_cols = [col for col in num_cols if col not in cat_cols]


    print(f"Observations: {dataframe.shape[0]}\nVariables: {dataframe.shape[1]}\ncat_cols: {len(cat_cols)}\n"
          f"num_cols: {len(num_cols)}\ncat_but_car: {len(cat_but_car)}\nnum_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# for döngüsünden sonra adult_male ve alone değişkenlerinin tipleri bool'dan int'e dönüştü


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    #print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
    #                   "Ratio": 100 * dataframe[col_name].value_counts(normalize=True)}))

    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)
