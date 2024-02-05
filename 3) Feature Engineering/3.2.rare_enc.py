#############################################
# Rare Encoding 
#############################################

# Rare Encoding: Bir df deki sınıfların countlarını aldığımızı düşünürsek, bu sınıfların sayıca az olanlarını encode etmek
# bize bir fayda sağlamayacaktır. Çünkü herhang bir vasıfları yoktur aslında. O yüzden rare olarak adlandıırıp df ye atarız.

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.


###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

from Functions.DataAnalysis import *


def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv")
    return data

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

# df deki kategorik değişkenleri seç.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getiren fonk.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)
