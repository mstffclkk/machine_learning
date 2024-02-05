from Functions.DataAnalysis import * # Functions/DataAnalysis.py Imports all functions in the file.


def load():
    data = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/titanic.csv") 
    return data

df = load()
df.head()

# df info
check_df(df)

# numerical ,categorical and cardinal variables
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# categorical variables summary
list(map(lambda x: cat_summary(df, x, plot=False), cat_cols))

# numerical variables summary
list(map(lambda x: num_summary(df, x, plot=False), num_cols))

# check outlier
for col in num_cols:
    print(col, check_outlier(df, col))

# outlier thresholds
for col in num_cols:
    print(col, outlier_thresholds(df, col))

# catch outlier index
for col in num_cols:
    print(col, "-->", grab_outliers(df, col, index=True))  

age_index = grab_outliers(df, "Age", index=True)
fare_index = grab_outliers(df, "Fare", index=True)

df.shape
###################################################################
# remove outlier
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
    new_df = remove_outlier(df, col)

new_df.shape

# or use drop function
new_dff = df.drop(fare_index, axis=0)
new_dff.shape

for col in num_cols:
    new_dff = df.drop(grab_outliers(df, col, index=True), axis=0)

new_dff.shape
###################################################################
# Re-assignment with thresholds

# check outlier
for col in num_cols:
    print(col, check_outlier(df, col))

# replace with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

# check outlier
for col in num_cols:
    print(col, check_outlier(df, col))

