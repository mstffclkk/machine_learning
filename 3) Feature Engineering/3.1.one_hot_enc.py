#############################################
# One-Hot Encoding
#############################################
# One Hot Encoderda, bir değişkenin tüm sınıfları tek tek değişkene dönüştürülür. Bu değişkenler tüm gözlem birimlerinde 0 ile doldurulur,
# fakat ilgili gözlemde, o gözlem birimi hangi sınıfa aitse, o sınıfın değişkeni 1 ile doldurulur.
# Burada dikkat edilmesi gerekilen bir husus var. One-Hot Encoding'i uygularken kullanacağımız metodlarda
# drop_first diyerek ilk sınıfı drop etmiş olarak ortaya çıkacak olan dummy değişken tuzağından kurtuluyoruz.

from Functions.DataAnalysis import *


def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df.head()
# embarked değişkeni 3 sınıftan oluşmakta ve bu sınıflar arasında herhangi bir seviye farkı yok.
df["Embarked"].value_counts() 

# get_dummies: df yi ve ilgili değişkeni ver, ben dönüştürürüm
pd.get_dummies(df, columns=["Embarked"], dtype=int).head()

# Sonrasında dummy değişken tuzağına düşmemek için drop_first parametresini kullanmamız gerekiyor
# drop_first=True, ilk sınıfı drop et. ilk sınıf alfabeye göre seçilir.
pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int).head()  


# Eksik değerleri de sınıf olarak getirmek.
# dummy_na=True, nan değerleri sınıfı oluşturulur. 
pd.get_dummies(df, columns=["Embarked"], dummy_na=True, dtype=int).head()    

# get_dummies metodunu kullnarak hem label encoding, hem de one hot encoding işlemini yapabiliriz.
# (LABEL ENCODING ICIN 2 SINIFLI OLMASI LAZIM)
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int).head()

#############   ONE HOT ENCODER FUNC    ##########

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Yukarıdaki fonksiyona kolonları verebilmem için encode etmek istediğimiz değişkenleri seçen bir list comprehension yapısı kullanırız:
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

df.head()
