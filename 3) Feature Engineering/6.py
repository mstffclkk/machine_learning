############################################# BÖLÜM 1 ###############################################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("/home/mustafa/PycharmProjects/pythonProject/github_repo/Miuul_Machine_Learning_Bootcamp/Datasets/application_train.csv")
    return data

df = load_application_train() 
df.head()


def load():
    data = pd.read_csv("/home/mustafa/PycharmProjects/pythonProject/github_repo/Miuul_Machine_Learning_Bootcamp/Datasets/titanic.csv")
    return data

df = load()
df.head()



#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler (Kutu grafik kullanılır(boxplot))
###################
# en yaygın kutu grafik, sonra histogram

sns.boxplot(x=df["Age"])  # bir sayısal değişkenin dağılım grafiğini verir
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################
# 1 - Eşik değerlere erişmek.
# Bir değişkenin çeyrek değerlerine erişmemiz lazım. Çünkü IQR (Inter Quartile Range) hesapliyoruz.

q1 = df["Age"].quantile(0.25)  # 1/4 (1. çeyrek)
q3 = df["Age"].quantile(0.75)  # 3/4 (3. çeyrek)
iqr = q3 - q1                  # Interquartile Range (IQR). 
up = q3 + 1.5 * iqr            # üst limit
low = q1 - 1.5 * iqr           # alt limit

df[(df["Age"] < low) | (df["Age"] > up)]            # aykırı değerleri yakalama.
df[~((df["Age"] < low) | (df["Age"] > up))]         # aykırı olmayan değerler.
df[(df["Age"] < low) | (df["Age"] > up)].index      # aykırı değerlerin indexlerini yakalama.

###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)  # axis=None -> herhangi bir satır veya sütun için çalışır.


df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

# eşik değerleri hesaplama fonksiyonu (kendisine girilen değerlerin eşik değerini hesaplar)

##################         OUTLIER THRESHOLDS FUNC       ##################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

# low, up değerlerini kaydetmek için.
low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()

df[(df["Fare"] < low) | (df["Fare"] > up)].index



##################         CHECK OUTLIER  FUNC       ##################
# Aykırı değer var mı yok mu bulan fonksiyon.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

###############################
##### grab_col_names  #########
###############################

dff = load_application_train()
dff.head()

# grab_col_names fonk
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal(çok yüksek sayıda değişkene sahip) değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    # cat_cols: Kategorik olanlar + Numerik gözüken ama kategorik olanlar - Kategorik gözüken ama kardinal olanlar
    # num_cols: Tipi object olmayanlar(int - float) - Numerik gözüken ama kategorik olanlar

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

########################################    titanic veri seri için   ##############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# num_cols PassengerId yi yakalamış. Bunun için num_cols listesinden çıkartıyoruz.
num_cols = [col for col in num_cols if col not in "PassengerId"] 

# nümerik değişkenler için check_outlier fonksiyonunu çağırıyoruz./
for col in num_cols:
    print(col, check_outlier(df, col))

########################################    application_train veri seri için   ##############################################

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# num_cols SK_ID_CURR yi yakalamış. Bunun için num_cols listesinden çıkartıyoruz.
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

# nümerik değişkenler için check_outlier fonksiyonunu çağırıyoruz./
for col in num_cols:
    print(col, check_outlier(dff, col))


###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

########################################    grab_outliers   ##############################################

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

# totalde 3 şey yaptık
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################
#aykırı değerleri silme
low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape #aykırı olmayan değerlerin sayısı

######## remove_outlier ##########
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

#########################################################
# Baskılama Yöntemi (re-assignment with thresholds)
#########################################################
# veri kaybetmemek için silmek yerine baskılama yapılır.

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]        # aykırı değerleri getirir.
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]    # aykırı değerleri getirir.(loc ile)


df.loc[(df["Fare"] > up), "Fare"]                           # üst sınıra göre aykırı değerler.
df.loc[(df["Fare"] > up), "Fare"] = up                      # üst sınıra göre aykırı değerleri üst sınıra(up) göre ayarla.
df.loc[(df["Fare"] < low), "Fare"] = low                    # alt sınıra göre aykırı değerleri alt sınıra(low) göre ayarla.

# programatik olarak yapmak.

######################### replace_with_thresholds ##############################
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# veri setini baştan yüklemek
df = load()  

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

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


# tek değişkenliler için yaptık.

###########################################################################################################################
#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor (LOF)
#############################################
# mülakat sorusu konuşuyo burda tekrar bak!!!!!!!!!!!!!
# 2 den fazla değişken 2 boyutta nasıl görselleştirilir. (40 ,50 100 değişkne)
# pci yöntemi ile yapılırmış.

## LOF = gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak aykırı değerleri belirleme. ##
# Bir noktanın lokal yoğunluğu ilgili noktanın etrafındaki komşuluklar demektir.
# Bir nokta komşularının noktalarından anlamlı bir şekilde düşükse, bu nokta daha seyrek bir bölgededir yani aykırı değer olabilir.
# LOF AYRICA THRESHOLD OLARAKTA YARDIM SAĞLAR

# 17 yaş , 3evlenme. tek başına aykırı olmyıp, çok değişkenli bir şekilde aykırı olabilir.

# diamonds veri setini, sadece sadece sayısal değişkenleri seçerek ve eksik değerleri drop ederek gettirmek.
df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()            # eksik ddeğerleri sil.
df.head()
df.shape

# aykırı değer var mı kontrol et ve yazdır.
for col in df.columns:
    print(col, check_outlier(df, col))

#########
# aykırı değişkenlerden sadece carat ı seçtik.
low, up = outlier_thresholds(df, "carat")

# carat değişkeninde kaç outlier var.
df[((df["carat"] < low) | (df["carat"] > up))]["carat"]        # aykırı değerleri getirir.
df[((df["carat"] < low) | (df["carat"] > up))].shape           # aykırı değerlerin sayısını getirir.

#########
# aykırı değişkenlerden sadece depth ı seçtik.
low, up = outlier_thresholds(df, "depth")

# depth değişkeninde kaç outlier var.
df[((df["depth"] < low) | (df["depth"] > up))].shape

# aykırı değerleri tek başına seçtiğimizde herbirinde çok fazla değere ulaştık.
# çok eğişkenli yaparsak ne olacak?
##########


clf = LocalOutlierFactor(n_neighbors=20)    # n_neighbors aranan komşuluk sayısıdır. ön tanımlı 20 olarak kullanılır.
clf.fit_predict(df)                         # LOF u veri setine uygulanır ve hesaplamalar yapılır.

# Lof değerlerini takip edebilmek için
df_scores = clf.negative_outlier_factor_    # skorları tutmamızı sağlayan bölüm.
df_scores[0:5]

# skorları pozitif değerlerle gözlemlemek isteyenler için.
# df_scores = -df_scores

# skorların negatif olması elbow tekniğinde daha kolay gözlem yapılır.

np.sort(df_scores)[0:5]                     # skorları sıralı(küçüten büyüğe) bir şekilde getir. (en kötü 5 skor)

# skorlarda pozitif değerler için 1 e yakın olanlar daha iyi, uzak olanlar daha kötü.
# skorlarda negatif değerler için -1 e yakın olanlar daha iyi, uzak olanlar daha kötü.


######################### her yerde bulunmaz ##################
# temel bileşen analizinde pci(?) da kullanılan elbow yöntemi.

# elbow yöntemi
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# threshold değeri
th = np.sort(df_scores)[3]          # 3. indeksteki skorun threshold değeri.(grafikten elde ettik)

df[df_scores < th]                  # threshold değerinden küçük skorları getir.(aykırı değerler seçmek için)

df[df_scores < th].shape            # aykırı değerlerin sayısını getir.


df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T  # yorumlama

df[df_scores < th].index

# silme işlemi
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# gözlem sayısı çok olduğunda baskılama yöntemi kullanmak çok mantıklı değildir.
# gözlem sayısı az olduğunda çok değişkenli baktıktan sonra o aykırı değişkenler silinebilir.

# ağaç yöntemleri kullanıyorsak aykırı değerlere hiç dokunmamayı tercih ederiz.
# illa dokunmak istiyorsak 1-99 5-95 gibi değerler kullanarak tıraşlama yapabiliriz.

# doğrusal yöntemler kullanıyorsak aykırı değerleri(az sayıdaysa) silmek mantıklıdır.
# ya da tek değişkenli yaklaşıp baskılama yapabiliriz.

#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

df.isnull().values.any()            # eksik gozlem var mı yok mu sorgusu
df.isnull().sum()                   # degiskenlerdeki toplam eksik deger sayisi
df.notnull().sum()                  # degiskenlerdeki tam deger sayisi
df.isnull().sum().sum()             # veri setindeki toplam eksik deger sayisi
df[df.isnull().any(axis=1)]         # en az bir tane eksik degere sahip olan gözlem birimleri
df[df.notnull().all(axis=1)]        # tam olan gözlem birimleri
df.isnull().sum().sort_values(ascending=False)   # Azalan şekilde sıralamak
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)    # eksik değişkenlerin tüm veri setine oranı
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]     # eksik değerlerin seçimi(isimlerinin yakalanması)



######################## eksik değerlerin fonksiyonu ##########################
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]                    # eksik değerlerin seçimi

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)                              # sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)  # oranı
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])     # birleştirme
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################
# Eksik Değer Problemini Çözme
#############################################
# eğer ağaca dayalı yöntemler kullanıılıyosa, missin values tıpkı outlier value ler gibi etkisi göz ardı edilebilir durumlardır.

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################

# gözlem sayısı çok fazla olduğunda 
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

###### SAYISAL DEĞİŞKENLERİ doldurmak için ######

# fillna() doldurma işlemi
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# df.apply(lambda x: x.fillna(x.mean()), axis=0)  # object old için hata veriyor.

###### NÜMERİK DEĞİKENLERİ fonksiyon kullanarak doldurmak için  ######
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head() 

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

###### KATEGORİK DEĞİŞKENELR için ######

# en mantıklı yöntem modunu alarak doldurmaktır.

df["Embarked"].mode()[0]    # modunu alıp ilk değere bak.

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")


###### KATEGORİK DEĞİŞKENELR için fonksiyon ######

# if tipi object ve unique değerleri 10 ve 10 dan küçükse mode ile doldurmaktır.
# else olduğu gibi kalsın
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

# cinsiyete göre titanic veri setini groupby a al.
# bu verilerin içinden yaş değişkenini seç ve ortalamasını al.
df.groupby("Sex")["Age"].mean()

df["Age"].mean()

"""
Amaç: Eksik değer doldurmak
    Yaş ortalaması:
    >>> df["Age"].mean() --> 29.69911764705882  

    Cinsiyete göre yaş ortalaması:
    >>> df.groupby("Sex")["Age"].mean()
    Sex
    female    27.915709
    male      30.726645

    Kadınlarda eksik değer varsa farklı, erkeklerde eksik değer varsa farklı doldurma işlemi
    yapmak, direk yaş ortalamasını alıp(29) ona göre eksik değer doldurmaktan daha mantıklıdır !!

"""

## fillna() metodu groupby dan gelecek kırılım ile ortalamaları cinsiyete göre doldurur !!
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


## bu işlemin loc karşılığı.

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female")]   # 1- yaş değişkeninde eksiklik olup cinsiyeti kadın olanlar.

df.groupby("Sex")["Age"].mean()["female"]              # 2- groupby kırılımında ki ortalamalardan kadınların ort seçmek.

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] # 3 - yaş değişkenini seçmiş olduk

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"] # 4- ve atama işlemi

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################
# makine öğrenmesi kullanılacak
# eksikliğe sahip değişken: bağımlı değişken
# diğer değişkenelr: bağımsız değişken

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# 2 sınıf veya daha fazla sınıfa sahip kategorik işlemleri nümerik şekilde ifade etmek.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

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

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
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




#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

# veri setinde olan değişkenlerdeki tam sayıları göstermektedir.
msno.bar(df)
plt.show()

# değişkenlerdeki eksikliklerin birlikte çıkıp çıkmadığıyla ilgili bilgi verir.
msno.matrix(df)
plt.show()

# eksiklikler üzerine kurulu bir ısı haritasıdır.
msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

# eksik değerlere sahip bütün değişkenleri getirmek.
missing_values_table(df, True)
na_cols = missing_values_table(df, True)

###### target değişken ile eksik değişkenleri karşılaştıran fonksiyon. ########
# eksik değere sahip: 1, dolu olanlar: 0

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        # eksik değerlere str ekle. Yani eksik değerler flagleniyor.
        # where(): eksik değerlere sahip değişkenleri 1, dolu değerleri 0 yapar.
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)       

    # temp df de seçim yap.[bütün satırları getir, bütün sütunlarda "_NA_" içeren sütunları seç] (buada list comp da kullanılabilir.)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        # eksik değerlerin kırılımını target ın mean i ile sağla
        # eksik değerlerin kırılımını target ın count ı ile sağla
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(), 
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

# amaç hayatta kalma durumunu etkileyen nedir ?

missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)




#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################
"""
elimizdeki kategorik değişken var. model kategorik değişkeni anlamayacağı için
numerik değişkene çevirmek lazım

label encoder 2 değerlilerde kullanılır.
2 den fazla değerli ise ordinal(sıralı) olması lazım. Yani 1.sınıf 2.sınıf ..
2 den fazla değerli olup nominal(sırasız) futbol takımları yani  ise one hot encoding

https://serdartafrali.medium.com/veri-biliminde-encoding-i%CC%87%C5%9Flemleri-616e87bf8c74
"""
# Encode: Değişkenlerin temsil şekillerinin değiştirilmesi.

#############################################
# Label Encoding & Binary Encoding
#############################################

# Bir kategorik değişkenin 2 sınıfı varsa bu 1-0 olarak kodlanırsa buna binary encoding denir.
# 2 den fazla sınıfı varsa label encoding denir.

df = load()
df.head()
df["Sex"].head()

# Label Encoder, değişken içerisindeki sınıfların alfabetik sırasına göre 0'dan başlayarak numaranlandırma yapar female -> 0, male -> 1
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]        # df değişkenini önce fit et sonra dönüştür.
le.inverse_transform([0, 1])            # 0,1 e karşılık gelen değerleri öğrenmek için.

# label encoder fonksiyonu.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

# Fonksiyonumuzu oluşturduk fakat bir problem var. Bir dataframe içerisinde yüzlerce değişken olduğu zaman,
# bunların arasından binary_col'ları (yani sadece 2 sınıf içeren kategorik değişkenleri) nasıl seçeriz?

# Bunun için de bu deeğişkenleri dataframeden ayıklamak için list comprehension kullanırız:

# df içerisindeki değişkenlerden tipi int ve float olmayan ve içerisinde 2 sınıf olan değişkenleri seç
binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]

# tüm binary_col larda gez ve çalıştır.
for col in binary_cols:
    label_encoder(df, col)

df.head()

## application_train dataseti için binary encoding yapılıyor.
df = load_application_train()
df.head()
df.shape

# 2 sınıflı kategorik değişkenler
binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]


df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()
# EMERGENCYSTATE_MODE değişkenindeki boş değerler 2 değeri ile dolduruldu,
# bunun farkında olduktan sonra bu işlem tehlikeli değildir, kimi zaman boş değerleri doldurmakta yöntem olarak da kullanılır.


df = load()
df["Embarked"].value_counts()       # embarked nominal değişken.
df["Embarked"].nunique()            # 3 değer var. (S, C, Q)
len(df["Embarked"].unique())        # 4 değer var. (S, C, Q, nan)

#############################################
# One-Hot Encoding
#############################################
# One Hot Encoderda, bir değişkenin tüm sınıfları tek tek değişkene dönüştürülür. Bu değişkenler tüm gözlem birimlerinde 0 ile doldurulur,
# fakat ilgili gözlemde, o gözlem birimi hangi sınıfa aitse, o sınıfın değişkeni 1 ile doldurulur.
# Burada dikkat edilmesi gerekilen bir husus var. One-Hot Encoding'i uygularken kullanacağımız metodlarda
# drop_first diyerek ilk sınıfı drop etmiş olarak ortaya çıkacak olan dummy değişken tuzağından kurtuluyoruz.

df = load()
df.head()
df["Embarked"].value_counts() # embarked değişkeni 3 sınıftan oluşmakta ve bu sınıflar arasında herhangi bir seviye farkı yok.

# get_dummies: df yi ve ilgili değişkeni ver, ben dönüştürürüm
pd.get_dummies(df, columns=["Embarked"]).head()

# Sonrasında dummy değişken tuzağına düşmemek için drop_first parametresini kullanmamız gerekiyor
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()  # drop_first=True, ilk sınıfı drop et. ilk sınıf alfabeye göre seçilir.

# Eksik değerleri de sınıf olarak getirmek.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()    # dummy_na=True, nan değerleri sınıfı oluşturulur. 

# get_dummies metodunu kullnarak hem label encoding, hem de one hot encoding işlemini yapabiliriz. (LABEL ENCODING ICIN 2 SINIFLI OLMASI LAZIM)
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

#############   ONE HOT ENCODER FUNC    ##########
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):  
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Yukarıdaki fonksiyona kolonları verebilmem için encode etmek istediğimiz değişkenleri seçen bir list comprehension yapısı kullanırız:
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()

df.head()

#############################################
# Rare Encoding 
#############################################

# Rare Encoding: Bir df deki sınıfların countlarını aldığımızı düşünürsek, bu sınıfların sayıca az olanlarını encode etmek
# bize bir fayda sağlamayacaktır. Çünkü herhang bir vasıfları yoktur aslında. O yüzden rare olarak adlandıırıp df ye atarız.

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.


###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

# df deki kategorik değişkenleri seç.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getiren fonk.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# bağımlı değişken ile kategroik değişkenin ilişkisini göstermek için bir fonk.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################
"""
    rare_perc: rare oranı.
    rare_columns: fonksiyona girilen rare oranın'dan daha düşük sayıda herhangi bir bu kategorik değişkenin sınıf oranı varsa
                  ve bu aynı zamanda bir kategorik değişken ise bunları rare kolonu olarak seç.
    temp_df[col].value_counts(): kategorik değişkenlerin frekansları, sayısı.
    len(temp_df): toplam gözlem sayısı.
    temp_df[col].value_counts() / len(temp_df): bu değişkenlerin oranı.
    any(axis=None): herhangi bir tanesi
""" 
def rare_encoder(dataframe, rare_perc):
    
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#############################################
# Feature Scaling (Özellik Ölçeklendirme)  
#############################################
""" 
1- Tüm değişkenleri eşit şartlar altında değerlendirebilmek adına ölçeklendirmektir.(Kullanılacak olan yöntemlere değişkenleri gönderirken onlara eşit muamele yapılması gerektiğini bildirmek için)
2- Özellikle gradient decent kullanan algoritmaların train sürelerini, yani eğitim sürelerini kısaltmak için.
3- Uzaklık temelli yöntemlerde yanlılığın önüne geçmek için

"""
###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s (yaygın)
###################
# Klasik standartlaştırma yöntemidir. Seçilen değerden, o değerin bulunduğu değişkenin ortalaması 
# çıkarılarak yine o değişkenin standart sapmasına bölünmesiyle hesaplanır. Z Standartlaştırması olarak da bilinir.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


################### ÖNEMLİ  ###################
# RobustScaler: Medyanı çıkar iqr'a böl.    
###################
"""
Robust scaler, standard scaler'a göre aykırı değerlere karşı dayanıklı
olduğundan dolayı daha tercih edilebilir olabilir. Fakat yaygın bir kullanım alanı yoktur. 
(vahit hoca standart scaler yerine daha kullanışlı old. düşünüyo)
"""
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü (yaygın)
###################
# Dönüştürmek istediğimiz özel bir alan varsa (mesela 1-5) kullanılabilir

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

# sonuçları karşılaştırmak için.(ortaya çıkan değerlerin yeni değişkenlerinde bir değişiklik var mı onu gözlemlemek)
age_cols = [col for col in df.columns if "Age" in col]

# num_summary: sayısal değişkenlerin çeyreklik değerlerini göstermek ve hist grafiğini oluşturmak
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################
# qcut metodu, bir değişkenin değerlerini küçükten büyüğe sıralar ve yazdığımız sayı kadar parçaya böler.
# hangi sınıflara dönüştürmek istediğimizi biliyorsak label eklenebilir.
# df["Age_qcut"] = pd.qcut(df['Age'], 5, labels=mylabels)

df["Age_qcut"] = pd.qcut(df['Age'], 5)
df.head()


#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################
"""
Feature extraction (özellik çıkarımı), ham veriden özellik (feature, değişken) üretmek. 2 çeşittir: 

1.  Yapısal verilerden değişken türetmek: var olan bazı değişkenler üzerinden yeni değişkenler türetmek
2.  Yapısal olmayan verilerden değişken üretmek: bilgisayarın anlayamayacağı metinsel veya görsel verilerden anlamlı değişkenler üretmek
"""
#############################################
# Binary Features: Flag, Bool, True-False
#############################################
"""
Var olan değişkenin içinden yeni değişkenler türetmek (1 / 0 şeklinde)

Dikkat! Yeni değişkenler üretmek istiyoruz, var olan değişkeni değiştirmek değil (bu encoding işlemi oluyor).
"""
df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int64')

#Şimdi bu yeni oluşturduğumuz değişkenin, bağımlı değişkene göre oranını inceleyelim
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

# Şimdi bu yeni oluşturduğumuz değişkenin, bağımlı değişkene göre oranını inceleyelim (yeni feature ile bağımlı değişken oranı)

from statsmodels.stats.proportion import proportions_ztest
"""
count: başarı sayısı
nobs: gözlem sayısı

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),   --> kabin numarası olan ve hayatta kalan kişi sayısı
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],  --> kabin numarası olmayan ve hayatta kalan kişi sayısı        

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0], --> kabin numarası olan kişi sayısı
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])--> kabin numarası olmayan kişi sayısı
"""

################## oran testi ##################
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
"""
Output: Test Stat = 9.4597, p-value = 0.0000
Proportion z testinin hipotezi (h0), p1 ve p2 oranları arasında fark yoktur der. p1 ve p2 oranları,
iki grubun, yani şu anki senaryoda cabin numarası olanların ve olmayanların hayatta kalma oranları.
İkisi arasında fark yoktur diyen hipotez, p-value değeri 0.05'ten küçük olduğundan dolayı geçersiz olmuş olur.
Yani bizim oluşturduğumuz değişkenden elde ettiğimiz oranların aralarında istatistiki olarak anlamlı bir farklılık var gibi gözüküyor.
(çünkü yukarıda p-value değerimiz 0 çıktı, <0.05)


Şimdi aynı şeyi başka bir değişken üzerinden yapalım.
Veriseti içerisinde SibSp ve Parch değişkenleri bulunuyor.
Bu değişkenler, o kişinin gemi içerisinde kaç tane yakını olduğunun bilgisini veren değişkenler.
Yeni bir değişken oluşturalım ve bu iki değişkenin toplamına göre, kişinin o teknede yalnız olup olmadığının bilgisini versin.
"""

# SibSp: yakın akrabalık, Parch: uzak akrabalık
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
"""
Output: `Test Stat = -6.0704, p-value = 0.0000`

p-value değerine bakıldığında h0 hipotezinin yine geçersiz olduğu görülür. Yani iki oran arasında istatistiki bir fark vardır.
"""
#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df = load()
df.head()

###################
# Letter Count
###################
# bir değişkende kaç tane harf var saydık.
df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################
# name içerisindeki metinleri split et ve içerisinde gez. dr ile başlayanları seç ve len ile sayısını bul.
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# dr ların hayatta kalma oranına bakalım.
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################
# ünvanları bulalım.

df.head()

# boşluk ile başlayıp nokta ile biten, ve büyük ve küçük harfler içeren ifadeleri yakala
# extract: çıkar
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# "NEW_TITLE", "Survived", "Age" 'i seç, "NEW_TITLE" a göre groupby a al .
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
"""
Normalde "Age" değişkeni içerisinde eksik değerler bulunuyor ve biz bunları genel olarak Age'in medyanı ile doldurabilirdik.
Fakat burada görüyoruz ki birçok title bulunuyor ve bunların hepsinin yaş ortalamaları farklı.
Dolayısıyla her bir eksik yaş verisini kendi title'ının ortalama yaşı ile doldurursak daha anlamlı bir veriseti oluşturmuş oluruz.
"""
#############################################
# Date Değişkenleri Üretmek
#############################################
# amaç: timestap üzerinden değişken üretmek.

dff = pd.read_csv("D:\\Users\\mstff\\PycharmProjects\\pythonProject\\datasets\\course_reviews.csv")
dff.head()
dff.info()

# problem: timestamp object tipinde. Timestamp değişkenini tipini değiğiştirmek gerek.

# dönüştürmek istediğin değişkeni ver ve değişken içerisindeki tarihlerin sıralanışına göre sırayı gir
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load()
df.head()

# Feature Interaction, değişkenlerin birbirleri ile etkileşime girmesi demektir.

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

# oluşturulan yeni featurelar bir şey ifade ediyor mu bakalım.
df.groupby("NEW_SEX_CAT")["Survived"].mean()


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################
# Amaç: insanların hayatta kalıp kalamayacağını titanic veri seti üzerinden modellemek.

df = load()
df.shape
df.head()

# bütün columları büyüttük.
df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Oluşturduğumuz bütün değişkenler.

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int64')

# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

# Ön işleme işlemleri yapmak için, hangileri sayısal kategorik seçmek lazım.

# Değişken isimlendirmelerini tutuyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Nümerik değişkende yer alan PASSENGERID yi kaldırıyoruz.
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler) # nümerik değişkenler için ön işleme basamağı
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

# kardinalitesi yüksek oldğu için
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# Oluşturmuş olduğumuz new_title'a göre groupby'a alıp yaş değişkeninin eksik değerlerini medyan ile doldur, new_title'a göre
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# AGE değişkenindeki eksiklikleri halletmiş olduk.
# Şimdi age değişkeninden türeyen değişkenlerdeki eksiklikleri de gidermek için bu değişkenleri tekrardan tanımlamak gerekiyor.
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# Sadece EMBARKED değişkeni kaldı. Bunu halletmek için de dataframe içerisinde tipi object olan,
# ve sınıf sayısı 10 veya daha altı olan değişkenlerdeki boşlukları, ilgili değişkenin modu ile dolduran bir fonksiyon yazacağız
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
df.head()

#############################################
# 4. Label Encoding
#############################################

# Bu kısımda iki sınıflı değişkenler için label encoding yöntemini kullanıyoruz.
# Öncelikle iki sınıflı değişkenleri seçmemiz gerekiyor.

# int ve float olmayan (yani kategorik olan) ve 2 sınıfa sahip olan değişkenleri seç
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64", "int32", "float32"]
               and df[col].nunique() == 2]

# Seçim işleminden sonra encoding işlemi
for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################
# Birleştirilmesi gereken sınıfları analiz edebilmemiz için rare_analyser fonksiyonumuzu kullanacağız.
rare_analyser(df, "SURVIVED", cat_cols)

# df içerisinde oranı 0.01 ve daha altı olan sınıfları birbirleri ile toplayıp yeni bir rare sınıfı içerisine atadık.
df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

# eşşiz değer sayısı 2den büyük ve 10dan küçük olan değişkenleri seç

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()
df.shape    

# Bu değişkenleri oluşturduk, fakat bu değişkenler gerekliler mi? Yani bir bilgi taşıyorlar mı yoksa taşımıyorlar mı?
# Bu sorunun cevabı için işlemde geriye gidip oluşturduğumuz yeni verisetinden değişkenleri tekrar ayıralım.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

"""
Burada yaptığımız işlemin sebebi, one hot encoderdan geçirdiğimiz ve 
yeni oluşan değişkenlerin hepsinin gerekli olup olmadığını bilmiyoruz bundan dolayı
bağımlı değişkenimize göre oranlarının ne olduğunu inceleyip işe yaramayan var mı onun analizini yaparız.

Yukarıdaki çıktı incelendiğinde bir sorun olduğunu görüyoruz; iki sınıflı olup sınıflarından
herhangi bir tanesinin oranı 0.01'den az olan var mı? Şimdi bu değişkenleri yakalayalım.
"""
############## ÖNEMLİ FONKSİYON ##############
# sınıf sayısı 2 olan ve value toplamlarının toplam verisetindeki satır sayısına oranı 0.01'den küçük olanları tut
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# SİLMEK İSTERSEK
# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler (öneri: robust scaler ya da min max scaler)
#############################################
# Bu senaryoda kullanacağımız modelden dolayı scale işlemi yapmamıza gerek kalmıyor. Fakat eğer yapacak olsaydık:

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################

y = df["SURVIVED"]                                  # bağımlı değişken
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)    # bağımsız değişkenler passengerid ve survived dışındaki değiişkenler

# train seti ve test seti oluştur
# train seti: model için kullanılacak set
# test seti: modelin test edileceği set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)



