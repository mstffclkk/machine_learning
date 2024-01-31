import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.width', 500)

pd.options.display.float_format = '{:.2f}'.format



"""
SaleId : Satış id
SaleDate : Satış Tarihi
CheckInDate : Müşterinin otelegiriş tarihi
Price : Satış için ödenen fiyat
ConceptName: Otel konsept bilgisi
SaleCityName: Otelin bulunduğu şehir bilgisi
CInDay: Müşterinin otele giriş günü
SaleCheckInDayDiff: Check in ile giriş tarihi gün farkı
Season: Otele giriş tarihindeki sezon bilgisi
"""

# Read the excel file
df = pd.read_excel('datasets/gezinomi.xlsx')

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

#[df[col].value_counts() for col in df.columns if df[col].nunique() < 10]

df["SaleCityName"].unique()
df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

df["ConceptName"].unique()
df["ConceptName"].nunique()
df["ConceptName"].value_counts()

df.groupby("SaleCityName").agg({"Price": ["sum", "mean"]})
df.groupby("ConceptName").agg({"Price": ["sum", "mean"]})
df.groupby(["SaleCityName", "ConceptName"]).agg({"Price":  "mean"})

# SaleCheckInDayDiff variable to categorical variable
bins = [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]
label = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]
df["EB_Scores"] = pd.cut(df["SaleCheckInDayDiff"], bins=bins, labels=label )

df.head()

df.groupby(["SaleCityName", "ConceptName", "EB_Scores"]).agg({"Price": ["mean", "count"]})
df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean", "count"]})
df.groupby(["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean", "count"]})

agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "sum"}).sort_values(("Price"), ascending=False)
agg_df.head()

agg_df.reset_index(inplace=True)
agg_df.head()


agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)

agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"]) 
agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})


agg_df.sort_values(by="Price")

new_user = "ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]







