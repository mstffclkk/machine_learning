#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################
from Functions.DataAnalysis import *

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

#df["NEW_NAME_DR"] = df["Name"].str.contains(" Dr. ")
#df["NEW_NAME_DR"].value_counts()
#df.loc[(df['NEW_NAME_DR'] == 1), ["Name", "NEW_NAME_DR"]]
#df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################
# ünvanları bulalım

df.head()

# boşluk ile başlayıp nokta ile biten, ve büyük ve küçük harfler içeren ifadeleri yakala
# extract: çıkar
df['NEW_TITLE'] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
df['NEW_TITLE'].value_counts()
df.loc[(df['NEW_TITLE'] == "Master"), :]
df.loc[(df['NEW_TITLE'] == "Rev"), :]

# veristeinde NEW_TITLE değişkeni içerisinde Dr. olan satırları seç ve göster.
df.loc[(df['NEW_TITLE'] == "Rev"), "NEW_TITLE"]


# "NEW_TITLE", "Survived", "Age" 'i seç, "NEW_TITLE" a göre groupby a al .
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": ["count", "mean"], "Age": ["count", "mean"]})

df.groupby(["NEW_TITLE"]).agg({"Survived": ["count", "mean"], 
                                "Age": ["count", "mean"]}).sort_values(by=("Survived", "count"), ascending=False)

           
