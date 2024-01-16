#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("/home/mustafa/PycharmProjects/pythonProject/github_repo/Miuul_Machine_Learning_Bootcamp/Datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1] # ilk ve son sütunu sil (id ve unnamed)
df.head()


num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

# korrelasyon: iki değişken arasındaki ilişkiyi ifade eder. -1 ile 1 arasında değer alır. 1'e yaklaştıkça pozitif,
# -1'e yaklaştıkça negatif ilişki vardır. 0 a yaklaştıkça ilişki zayıflar.
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show() 




#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#######################

cor_matrix = df[num_cols].corr().abs()

#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000


#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN


# üst üçgen matrisi
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# herhangi bir sütunda 0.90'dan büyük değer varsa o sütunu seç.
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

# yüksek korelasyonlu değişkenlerin listesi
cor_matrix[drop_list]

# yüksek korelasyonlu değişkenlerin silinmesi
df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    dataframe = dataframe.iloc[:, 1:-1] # dataya göre kullanılmayabilir.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in [int, float]]
    corr = df[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)

# yüksek korelasyonlu değişkenlerin silinmesi
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

# Yaklaşık 600 mb'lık 300'den fazla değişkenin olduğu bir veri setinde deneyelim.
# https://www.kaggle.com/c/ieee-fraud-detection/data?select=train_transaction.csv

df = pd.read_csv("datasets/fraud_train_transaction.csv")
len(df.columns)
df.head()

drop_list = high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)