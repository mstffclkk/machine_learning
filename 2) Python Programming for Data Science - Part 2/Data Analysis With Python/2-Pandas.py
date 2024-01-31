#############################################
# PANDAS
#############################################

# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look at Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri

#############################################
# Pandas Series
#############################################
# - pandas series: tek boyutlu ve index bilgisi içeren bir veri tipidir(array).
# - pandas dataframe: çok boyutlu ve index bilgisi içeren bir veri tipidir(tablo).
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

#############################################
# Veri Okuma (Reading Data)
#############################################
import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head()

#############################################
# Veriye Hızlı Bakış (Quick Look at Data)
#############################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()                   # ilk 5 gözlem
df.tail()                   # son 5 gözlem
df.shape                    # boyut bilgisi
df.info()                   # veri seti hakkında bilgi
df.axes                     # satır ve sütunlar hakkında bilgi
df.columns                  # değişken isimleri
df.index                    # index bilgisi
df.describe().T             # betimsel istatistikler
df.isnull().values.any()    # herhangi bir eksik gözlem var mı?
df.isnull().sum()           # her bir değişkende kaçar tane eksik gözlem var? 
df.isnull().sum()[df.isnull().sum() != 0] # her bir değişkende (0 olmayanlar hariç) eksik gözlemleri göster. 

df["sex"].head() 
df["sex"].value_counts()    # kategorik değişken sınıflarının frekansları
df["embarked"].value_counts()

#############################################
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
#############################################
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(5, axis=0).head() # axis=0 -> satır bazında işlem yapılacağını belirtir. (5. indeksteki satırı sil)

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# değişikliklerin kalıcı olması için inplace=True.
# df = df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)

#######################
# Değişkeni Indexe Çevirmek
#######################

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head()

#######################
# Indexi Değişkene Çevirmek
#######################

df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head() # indexi değişkene çevirirken eski indexi de değişken olarak ekler.
df = df.reset_index()
df.head()

#######################
# Değişkenler (sütun) Üzerinde İşlemler
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()
type(df["age"].head())

df[["age"]].head()          # "age" değişkenini seçmek için 2 tane çift köşeli parantez kullanılır.
type(df[["age"]].head())    # 2 boyutlu bir dataframe döndürür.

df[["age", "alive"]]        # birden fazla değişken seçimi

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()


#######################
# iloc & loc
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection (indexlerden seçim yapar e kadar)
df.iloc[0:3]  # 0, 1, 2. satırlar
df.iloc[0, 0] # 0. satır, 0. sütun
 
# loc: label based selection (indexlerden seçim yapar)
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]


#######################
# Koşullu Seçim (Conditional Selection)
#######################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()   
df[df["age"] > 50]["age"].count()       # ["age"] i belirtmemiz lazım. 50 yaşından büyük olanların sayısı

df.loc[df["age"] > 50, ["age", "class"]].head() # 50 yaşından büyük olanların yaş ve class bilgileri. (bir koşul, 2 sütun) seçildi

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head() # 50 yaşından büyük erkeklerin yaş ve class bilgileri.

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

###################################################################################################
# Toplulaştırma ve Gruplama (Aggregation & Grouping) !!!!!!!!!!! DİKKAT !!!!!!!!!!!
###################################################################################################

# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()                                    # "age" değişkeninin ortalaması

df.groupby("sex")["age"].mean()                     # "sex" değişkenine göre "age" değişkeninin ortalaması. yani cinsiyete göre yaş ortalaması

# bu kullanım önerilir.
df.groupby("sex").agg({"age": "mean"})              # yukarıdaki ile aynı sonucu verir. 
df.groupby("sex").agg({"age": ["mean", "sum"]})     # cinsiyete göre yaş ortalaması ve toplamı
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})


##################################################################################
# Pivot table   (df, values=[], index=[], aggfunc={})
##################################################################################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# aşağıda groupby ile ve pivot_table ile aynı sonucu elde ediyoruz.
df.groupby(["sex", "embark_town"]).agg({"survived": "mean"})

df.pivot_table("survived", "sex", "embarked", aggfunc="mean")   # sex'e göre, embarked'a göre survived ortalaması

# aşağıda groupby ile ve pivot_table ile aynı sonucu elde ediyoruz.
df.groupby(["sex", "embarked", "class"]).agg({"survived": "mean"})

df.pivot_table("survived", "sex", ["embarked", "class"], aggfunc="mean")        # sex'e göre, embarked ve class'a göre survived ortalaması


# pd.cut() fonksiyonu ile yaş değişkenini kategorik değişkene çeviriyoruz.
# pd.cut(): belirtilen aralıklara göre kategorik değişken oluşturur.
# pd.qcut(): ise belirtilen aralıklara göre kategorik değişken oluştururken, aralıkların eşit olmasına dikkat eder.

# pd.cut(): eğer aralıkları biliyorsak kullanılır.
# pd.qcut(): eğer çeyreklik değerlere görebölmek itiyorsak kullanılır.


df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


#############################################
# Apply ve Lambda
#############################################
# apply(): satır veya sütunlarda otomatik olarak fonksiyon çalıştırmamızı sağlar.
# lambda: fonksiyon oluşturmadan fonksiyon kullanmamızı sağlar.(kullan at fonksiyon)

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head() # "age" içeren tüm sütunları seç ve 10'a böl

# standartlaştırma
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# kaydetmek için
# df.loc[:, ["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
# df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()
#############################################
# Birleştirme (Join) İşlemleri 
#############################################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

# pd.concat(): axis=0 satır bazında birleştirme yapar. axis=1 sütun bazında birleştirme yapar.
pd.concat([df1, df2]) # satır bazında birleştirme
 
pd.concat([df1, df2], ignore_index=True) # indexleri düzeltme

#######################
# Merge ile Birleştirme İşlemleri
#######################

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)

