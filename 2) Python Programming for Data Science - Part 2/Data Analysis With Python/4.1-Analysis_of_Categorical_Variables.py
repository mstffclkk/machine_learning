
#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()  # sınıf frekansları
df["sex"].unique()  # sınıf isimleri
df["sex"].nunique()  # sınıf sayısı


[df[col].value_counts() for col in df.columns]
[df[col].unique() for col in df.columns]
[df[col].nunique() for col in df.columns]

# CATEGORICAL = olası kategorik değişkenlerin tip bilgisine göre tespiti
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# CATEGORICAL = numerik gözüken ama aslında kategorik olan değişkenlerin tespiti (10 sınıf eşik değeri projeye göre değişebilir)
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64", "int32", "float32"]]

# CARDINAL = kategorik gözüken ama aslında kardinal(çok fazla sınıfı olan) olan değişkenlerin tespiti (20 sınıf eşik değeri projeye göre değişebilir)
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

# CATEGORICAL
cat_cols = cat_cols + num_but_cat

# CATEGORICAL = kardinal değişkenlerinin içinde yoksa kategoriktir. (cat_cols - cat_but_car)
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

# kategorik olmayanlar yani nümerik
num_cols = [col for col in df.columns if col not in cat_cols]


# df["survived"].value_counts()
# 100 * df["survived"].value_counts() / len(df)

########################################
## cat_summary fonksiyonu
########################################

# cat_summary fonksiyonu ile tüm kategorik değişkenlerin sınıf frekanslarına ve yüzdeliklerine bakalım
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("\n##########################################")


cat_summary(df, "sex")

# bulduğumuz kategorik değişkenlere fonksiyonu uygulayalım
for col in cat_cols:
    cat_summary(df, col)

list(map(lambda x: cat_summary(df, x), cat_cols))


# cat_summary plot update
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("type is bool !!\n")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)

# yukarıdaki for döngüsünü list map ve lambda ile yapalım
list(map(lambda x: cat_summary(df, x, plot=True), cat_cols))


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


cat_summary(df, "adult_male", plot=True)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


cat_summary(df, "sex")