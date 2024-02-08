#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
# Feature Interaction, değişkenlerin birbirleri ile etkileşime girmesi demektir.
# 2 değişkenin çarpımı, bölümü, karesi, küpü gibi işlemlerle yeni değişkenler üretilebilir.

from Functions.DataAnalysis import *

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data
df = load() 
df.head()


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
