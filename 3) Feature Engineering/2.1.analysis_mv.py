
from Functions.DataAnalysis import *

def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load()
df.head()
check_df(df)


#############################################
# Advanced Analytics
#############################################

###################
# Examining the Missing Data Structure
###################

msno.bar(df) # veri setindeki tam gözlem sayısını gösterir.
plt.show()

msno.matrix(df) # downgrade matplotlib to 3.6 or upgrade missingno to 0.5.2
plt.show()

msno.heatmap(df)
plt.show()

###################
# Examining the Relationship of Missing Values with Dependent Variable
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)

missing_vs_target(df, "Survived", na_cols)



###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)

df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

missing_vs_target(df, "Survived", na_cols)


