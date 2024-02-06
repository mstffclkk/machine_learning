#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################
 
#############################################
# Binary Features: Flag, Bool, True-False
#############################################
# Var olan değişkenin içinden 1-0 şeklinde yeni değişkenler türetmek

from Functions.DataAnalysis import *

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df.head()

#########################################################3
# binarize edilmiş bir değişken oluşturduk.
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

# yeni feature ile bağımlı değişken arasındaki ilişki.
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

# count: başarı sayısı
# nobs: gözlem sayısı

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),    # kabin numarası olan ve hayatta kalan kişi sayısı
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],   # kabin numarası olmayan ve hayatta kalan kişi sayısı        

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],  # kabin numarası olan kişi sayısı
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]]) # kabin numarası olmayan kişi sayısı

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#########################################################
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
  