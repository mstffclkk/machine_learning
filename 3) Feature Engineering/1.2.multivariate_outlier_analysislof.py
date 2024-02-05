#############################################
# Multivariate Outlier Analysis: Local Outlier Factor (LOF)
#############################################
from Functions.DataAnalysis import * 

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64']) # nümerik değişkenleri seçtik.
df = df.dropna()            # eksik değerleri sil.
df.head()

check_df(df)

for col in df.columns:
    print(col, check_outlier(df, col))

for col in df.columns:
    print(col, outlier_thresholds(df, col))

###################################################################################################
# aykırı değişkenlerden sadece carat ı seçtik.
low, up = outlier_thresholds(df, "carat")

# carat değişkeninde kaç outlier var.
df[((df["carat"] < low) | (df["carat"] > up))]["carat"]       # aykırı değerleri getirir.
df[((df["carat"] < low) | (df["carat"] > up))].shape           # aykırı değerlerin sayısını getirir.

###################################################################################################
# aykırı değişkenlerden sadece depth ı seçtik.
low, up = outlier_thresholds(df, "depth")

# depth değişkeninde kaç outlier var.
df[((df["depth"] < low) | (df["depth"] > up))].shape

# aykırı değerleri tek başına seçtiğimizde herbirinde çok fazla değere ulaştık.
# çok değişkenli yaparsak ne olacak?
###################################################################################################

for col in df.columns:
    check_outlier(df, col)

############### LOF ###############

clf = LocalOutlierFactor(n_neighbors=20)    # n_neighbors aranan komşuluk sayısıdır. ön tanımlı 20 olarak kullanılır.
clf.fit_predict(df)                         # LOF u veri setine uygulanır ve hesaplamalar yapılır.

# Lof değerlerini takip edebilmek için
df_scores = clf.negative_outlier_factor_    # skorları tutmamızı sağlayan bölüm.
df_scores[0:5]
# skorları pozitif değerlerle gözlemlemek istersek.
# df_scores = -df_scores

# skorların negatif olması elbow tekniğinde daha kolay gözlem yapılır. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

np.sort(df_scores)[0:5]                     # skorları sıralı(küçüten büyüğe) bir şekilde getir. (en kötü 5 skor)

# skorlarda pozitif değerler için 1 e yakın olanlar daha iyi, uzak olanlar daha kötü.
# skorlarda negatif değerler için -1 e yakın olanlar daha iyi, uzak olanlar daha kötü.

######################### önemli ##################
# temel bileşen analizinde pci(?) da kullanılan elbow yöntemi.

# elbow yöntemi
# grafikteki dirsek noktasını eğimi en dik olan nokta olarak seçiyoruz.
# bu da bizim threshold değerimiz oluyor.
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# threshold değeri
th = np.sort(df_scores)[3]          # 3. indeksteki skorun threshold değeri.(grafikten elde ettik)
df[df_scores < th]                  # threshold değerinden küçük skorları getir.(aykırı değerler seçmek için)
df[df_scores < th].shape            # aykırı değerlerin sayısını getir.


# neden 3 tane aykırı değer var? incelemek için
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T  # yorumlama


# peki aykırı değerleri ne yapacağız?

# indexlerini alabiliriz
df[df_scores < th].index

# silebiliriz
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# gözlem sayısı çok olduğunda baskılama yöntemi kullanmak çok mantıklı değildir.
# gözlem sayısı az olduğunda çok değişkenli baktıktan sonra o aykırı değişkenler silinebilir.

# ağaç yöntemleri kullanıyorsak aykırı değerlere hiç dokunmamayı tercih ederiz.
# illa dokunmak istiyorsak 1-99 5-95 gibi değerler kullanarak tıraşlama yapabiliriz.
# iqr kullanarak tıraşlama yapabiliriz. (outlier_thresholds fonksiyonu )

# doğrusal yöntemler kullanıyorsak aykırı değerleri(az sayıdaysa) silmek mantıklıdır.
# aykırı değerleri de doldurmak yerine de tek değişkenli yaklaşıp ucundan baskılama yapabiliriz.










