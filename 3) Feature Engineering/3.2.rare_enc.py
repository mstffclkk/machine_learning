#############################################
# Rare Encoding 
#############################################

# Rare Encoding: Bir df deki sınıfların countlarını aldığımızı düşünürsek, bu sınıfların sayıca az olanlarını encode etmek
# bize bir fayda sağlamayacaktır. Çünkü herhangi bir vasıfları yoktur aslında. O yüzden rare olarak adlandırıp df ye atarız.

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazılması.


###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

from Functions.DataAnalysis import *

def load_application_train():
    data = pd.read_csv("/home/mustafa/github_repo/dataset/application_train.csv")
    return data

df = load_application_train()
df.head()
check_df(df)

df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder
#############################################
"""
    rare_perc: rare orani.
    rare_columns: fonksiyona girilen rare oranin'dan daha düşük sayida herhangi bir bu kategorik değişkenin sinif orani varsa
                  ve bu ayni zamanda bir kategorik değişken ise bunlari rare kolonu olarak seç.
    temp_df[col].value_counts(): kategorik değişkenlerin frekanslari, sayisi.
    len(temp_df): toplam gözlem sayisi.
    temp_df[col].value_counts() / len(temp_df): bu değişkenlerin orani.
    any(axis=None): herhangi bir tanesi
""" 
def rare_encoder(dataframe, rare_perc):
    
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)
