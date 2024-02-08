#############################################
# Date Değişkenleri Üretmek
#############################################
# amaç: timestap üzerinden değişken üretmek.

dff = pd.read_csv("/home/mustafa/github_repo/machine_learning/datasets/course_reviews.csv")
dff.head()
dff.info()

# problem: timestamp object tipinde. Timestamp değişkenini tipini değiğiştirmek gerek.

# dönüştürmek istediğin değişkeni ver ve değişken içerisindeki tarihlerin sıralanışına göre sırayı gir
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date
