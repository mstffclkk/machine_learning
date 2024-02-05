#############################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################
from Functions.DataAnalysis import *


def load_application_train():
    data = pd.read_csv("/home/mustafa/github_repo/dataset/application_train.csv")
    return data

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df = load_application_train()

#############################################
# Label Encoding & Binary Encoding(0-1 Encoding)
#############################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5] # fit_transform: uygula ve dönüştür. Alfabetik sıraya göre 0 ve 1'e dönüştürdü.
le.inverse_transform([0, 1])     # inverse_transform: dönüştürülmüş hali geri dönüştür.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()
df.loc[:, binary_cols]

# nunique eksik değerleri saymaz. Eğer eksik değerler varsa, eksik değerleri dikkate almadan sadece unique değerleri sayar.
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

#############################################
# load_application_train dataset

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df.head()
df.loc[:, binary_cols]
