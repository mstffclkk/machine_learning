###################################################################
# Libraries
###################################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################################################################
# Info of DataFrame
###################################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("\n##################### Info #####################")
    print(dataframe.info())
    print("\n##################### Types #####################")
    print(dataframe.dtypes)
    print("\n##################### Head #####################")
    print(dataframe.head(head))
    print("\n##################### Tail #####################")
    print(dataframe.tail(head))
    print("\n################ Any Null Values ##################")
    print(dataframe.isnull().values.any())
    print("\n##################### NA Sum #####################")
    print(dataframe.isnull().sum())
    print("\n##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("\n##################### Categorical Value Counts #####################")
    print([dataframe[col].value_counts() for col in dataframe.columns if dataframe[col].nunique() < 10])

###################################################################
# Capture of Numerical and Categorical Variables
###################################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
     It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
     Note: Categorical variables with numerical appearance are also included.

     parameters
     ------
         dataframe: dataframe
                 Dataframe from which variable names are to be taken
         cat_th: int, optional
                 Class threshold value for variables that are numeric but categorical
         car_th: int, optional
                 class threshold for categorical but cardinal variables

     returns
     ------
         cat_cols: list
                 Categorical variable list
         num_cols: list
                 Numerical variable list
         cat_but_car: list
                 List of cardinal variables with categorical view

     examples
     ------
         import seaborn as sns
         df = sns.load_dataset("iris")
         print(grab_col_names(df))

         cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
         print(f"\ncat_cols: {cat_cols}\nnum_cols: {num_cols}\ncat_but_car: {cat_but_car}")

     Notes
     ------
         cat_cols + num_cols + cat_but_car = total number of variables
         num_but_cat is inside cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}\nVariables: {dataframe.shape[1]}\ncat_cols: {len(cat_cols)}\n"
          f"num_cols: {len(num_cols)}\ncat_but_car: {len(cat_but_car)}\nnum_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def grab_col_name(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                     dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                        dataframe[col].dtypes == "O"]                           
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}\nVariables: {dataframe.shape[1]}\ncat_cols: {len(cat_cols)}\n"
            f"num_cols: {len(num_cols)}\ncat_but_car: {len(cat_but_car)}\nnum_but_cat: {len(num_but_cat)}")                                         
    return cat_cols, num_cols, cat_but_car


###################################################################
# Analysis of Categorical Variables (cat_summary)
###################################################################
"""
If there are bool values, we convert them to int.

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

"""

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

"""
for col in cat_cols:
    cat_summary(df, col, plot=True)

"""

###################################################################
# Analysis of Numerical Variables num_summary
###################################################################
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
        
        
"""
for col in num_cols:
    num_summary(df, col, plot=True)
"""

###################################################################
# ANALYSIS OF TARGET VARIABLE
###################################################################
## For Categorical Variables

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

"""for col in cat_cols:
    target_summary_with_cat(df, "TARGET", col)
"""
## For Numerical Variables
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
"""
for col in num_cols:
    target_summary_with_num(df, "TARGET", col)"""

####################### Data Preprocessing ########################

###################################################################
# Outlier Detection Process
###################################################################
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper limits for identifying outliers in a given column of a dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the column.
    col_name (str): The name of the column.
    q1 (float, optional): The lower quartile value. Defaults to 0.25.
    q3 (float, optional): The upper quartile value. Defaults to 0.75.

    Returns:
    tuple: A tuple containing the lower and upper limits for identifying outliers.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

"""

# numeric col
for col in num_cols:
    print(col, "-->", outlier_thresholds(df, col))

# categoric col
for col in cat_cols:
    print(col, "-->", outlier_thresholds(df, col))

"""

###################################################################
# Outlier Control by Thresholds
###################################################################
#def check_outlier(dataframe, col_name):
#    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
#    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
#        return True
#    else:
#        return False

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].any(axis=None):
        outlier_df = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]
        print(f"{col_name}: {outlier_df.shape[0]} outlier(s) found.")
        return True
    return False

"""
for col in num_cols:
    print(col, "-->", check_outlier(df, col))
"""

###################################################################
# To See the Available Outliers
###################################################################
def grab_outliers(dataframe, col_name, index=False, h=5):
    """
    Prints or returns the outliers in a given column of a dataframe.

    Parameters:
    - dataframe (pandas.DataFrame): The dataframe containing the data.
    - col_name (str): The name of the column to analyze. (for numerical columns)
    - index (bool, optional): Whether to return the indices of the outliers. Default is False.
    - h (int, optional): The number of rows to display when printing outliers. Default is 5.

    Returns:
    - outlier_index (pandas.Index): The indices of the outliers, if index=True. Otherwise, None.
    """
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(h))
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

"""
for col in num_cols:
    print(col, "-->", grab_outliers(df, col, True))  

"""

###################################################################
# Re-assignment with thresholds
###################################################################
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


"""
# Outlier Analysis and Suppression Process
for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))
"""

###################################################################
# Missing Value Observation
###################################################################
def missing_values_table(dataframe, na_name=False):
    """
    Prints a table showing the number and percentage of missing values in each column of the given dataframe.

    Parameters:
    - dataframe: The pandas DataFrame to analyze.
    - na_name (optional): If True, returns a list of column names with missing values.

    Returns:
    - If na_name is True, returns a list of column names with missing values.
    - Otherwise, prints the missing values table.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

###########################################################################
# Examining the Relationship of Missing Values ​​with the Dependent Variable
###########################################################################
def missing_vs_target(dataframe, target, na_columns):
    """
    Calculate the mean target value and count of each category in na_columns.

    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    target (str): The name of the target column.
    na_columns (list): List of column names to analyze.

    Returns:
    None
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), "NA", "not NA")
    na_flags = [col for col in temp_df.columns if "_NA_" in col]
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

###################################################################
# ENCODING
###################################################################

###################################################################
## LABEL ENCODING
###################################################################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

"""
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)
"""

###################################################################
## ONE - HOT ENCODING
###################################################################
# cat_cols listesinin güncelleme işlemi
"""
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols
"""
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

###################################################################
## Rare Analyser
###################################################################
# bağımlı değişken ile kategroik değişkenin ilişkisini göstermek için bir fonk.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

###################################################################
## RARE ENCODING
###################################################################
"""
    rare_perc: rare orani.
    rare_columns: fonksiyona girilen rare oranin'dan daha düşük sayida herhangi bir bu kategorik değişkenin sinif 
                  orani varsa ve bu ayni zamanda bir kategorik değişken ise bunlari rare kolonu olarak seç.
    temp_df[col].value_counts(): kategorik değişkenlerin frekanslari, sayisi.
    len(temp_df): toplam gözlem sayisi.
    temp_df[col].value_counts() / len(temp_df): bu değişkenlerin orani.
    any(axis=None): herhangi bir tanesi

    usage:
    new_df = rare_encoder(df, 0.01)
    rare_analyser(new_df, "TARGET", cat_cols)

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

###################################################################
"""# Correlation
###################################################################
df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
"""

###################################################################
# Base Model
###################################################################

"""y = df["TARGET"]
X = df.drop("TARGET", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")
"""

###################################################################
# FEATURE IMPORTANCE
###################################################################
"""def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)"""