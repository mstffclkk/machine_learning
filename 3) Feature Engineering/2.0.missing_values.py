#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################
from Functions.DataAnalysis import *


def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df.head()
check_df(df)

df.isnull().values.any()            # eksik gozlem var mı yok mu sorgusu
df.isnull().sum()                   # degiskenlerdeki toplam eksik deger sayisi
df.isnull().sum().sort_values(ascending=False)   # Azalan şekilde sıralamak
df.isnull().sum().sum()             # veri setindeki toplam eksik deger sayisi (kendisinde en az 1 tane eksik deger olan satir sayisi)
df[df.isnull().any(axis=1)]         # en az bir tane eksik degere sahip olan gözlem birimleri
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)    # eksik değişkenlerin tüm veri setine oranı
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]     # eksik değerlerin olduğu değişkenler

df.notnull().sum()                  # degiskenlerdeki tam deger sayisi
df[df.notnull().all(axis=1)]        # tam olan gözlem birimleri

###################################################################################
# Eksik Değer Problemini Çözme
###################################################################################
# eğer ağaca dayalı yöntemler kullanıılıyosa, eksik değerler
# tıpkı aykırı değerlerdeki gibi etkisi göz ardı edilebilir durumlardır.

missing_values_table(df)
missing_values_table(df, True)

############################################################################
# Çözüm 1: Hızlıca silmek
############################################################################
# gözlem sayısı çok fazla olduğunda kullanılabilir.
df.dropna().shape

############################################################################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
############################################################################

###### SAYISAL DEĞİŞKENLERİ doldurmak için ######

# fillna() doldurma işlemi
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)  
# kategorik değişkenler de olduğu için hata veriyor.

###### SAYISAL DEĞİŞKENLERİ fonksiyon kullanarak doldurmak için  ######
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)

############################################################################

###### KATEGORİK DEĞİŞKENELR için ######
# en mantıklı yöntem modunu alarak doldurmaktır.

df["Embarked"].mode()[0]    # modunu alıp ilk değere bak.
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing")

###### KATEGORİK DEĞİŞKENELR için fonksiyon ######

# if tipi object ve unique değerleri 10 ve 10 dan küçükse mode ile doldurmaktır.
# else olduğu gibi kalsın
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

############################################################################


#########################################################
# Kategorik Değişken Kırılımında Değer Atama
#########################################################

# cinsiyete göre titanic veri setini groupby a al.
# bu verilerin içinden yaş değişkenini seç ve ortalamasını al.
df.groupby("Sex")["Age"].mean()
df.groupby("Sex").agg({"age": "mean"})

df["Age"].mean()

"""
Amaç: Eksik değer doldurmak
    Yaş ortalamasi:
    >>> df["Age"].mean() --> 29.69911764705882  

    Cinsiyete göre yaş ortalamasi:
    >>> df.groupby("Sex")["Age"].mean()
    Sex
    female    27.915709
    male      30.726645

    Kadinlarda eksik değer varsa farkli, erkeklerde eksik değer varsa farkli doldurma işlemi
    yapmak, direk yaş ortalamasini alip(29) ona göre eksik değer doldurmaktan daha mantiklidir !!

"""

## fillna() metodu groupby dan gelecek kırılım ile ortalamaları cinsiyete göre doldurur !!
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

## bu işlemin loc karşılığı.

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female")]                                                    # 1- yaş değişkeninde eksiklik olup cinsiyeti kadın olanlar.
df.groupby("Sex")["Age"].mean()["female"]                                                               # 2- groupby kırılımında ki ortalamalardan kadınların ort seçmek.
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"]                                             # 3 - yaş değişkenini seçmiş olduk
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"] # 4- ve atama işlemi

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull().sum()

#######################################################################################################################################
##########################################################################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
##########################################################################################
# makine öğrenmesi kullanılacak
# eksikliğe sahip değişken: bağımlı değişken
# diğer değişkenelr: bağımsız değişken

df = load()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# 2 sınıf veya daha fazla sınıfa sahip kategorik işlemleri nümerik şekilde ifade etmek.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True, dtype=int)
dff.head()

# değişkenlerin standartşatırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# buraya kadar ki işlemler makine öğrenmesi tekniğini kullanmak için veriyi hazır hale getirmekti.


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# standartlaştırılan değerleri geri alma.
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# doldurulan değerleri karşılaştırmak için(nereye ne atandı)
df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]].head()
df.loc[df["Age"].isnull()]



###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
















































