
###################
# Regex ile Değişken Türetmek
###################
# ünvanları bulalım.

df.head()

# boşluk ile başlayıp nokta ile biten, ve büyük ve küçük harfler içeren ifadeleri yakala
# extract: çıkar
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# "NEW_TITLE", "Survived", "Age" 'i seç, "NEW_TITLE" a göre groupby a al .
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
"""
Normalde "Age" değişkeni içerisinde eksik değerler bulunuyor ve biz bunları genel olarak Age'in medyanı ile doldurabilirdik.
Fakat burada görüyoruz ki birçok title bulunuyor ve bunların hepsinin yaş ortalamaları farklı.
Dolayısıyla her bir eksik yaş verisini kendi title'ının ortalama yaşı ile doldurursak daha anlamlı bir veriseti oluşturmuş oluruz.
"""