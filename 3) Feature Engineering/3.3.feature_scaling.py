#############################################
# Feature Scaling (Özellik Ölçeklendirme)  
#############################################
""" 
1- Tüm değişkenleri eşit şartlar altinda değerlendirebilmek adina ölçeklendirmektir.
  (Kullanilacak olan yöntemlere değişkenleri gönderirken onlara eşit muamele yapilmasi gerektiğini bildirmek için)
2- Özellikle gradient decent kullanan algoritmalarin train sürelerini, yani eğitim sürelerini kisaltmak için.
3- Uzaklik temelli yöntemlerde yanliliğin önüne geçmek için

"""
###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s (yaygın)
###################
# Klasik standartlaştırma yöntemidir. Seçilen değerden, o değerin bulunduğu değişkenin ortalaması 
# çıkarılarak yine o değişkenin standart sapmasına bölünmesiyle hesaplanır. Z Standartlaştırması olarak da bilinir.

from Functions.DataAnalysis import *

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

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

