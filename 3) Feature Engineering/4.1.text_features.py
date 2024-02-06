
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
