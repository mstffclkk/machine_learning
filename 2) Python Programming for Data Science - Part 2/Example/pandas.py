"""
---------------------------------------------------
|                Pandas Cheatsheet                |
---------------------------------------------------

Bu cheatsheet, Pandas'ın temel işlemlerini kapsamaktadır. Pandas'ın daha gelişmiş ve detaylı işlevlerini keşfetmek için resmi Pandas dokümantasyonuna başvurmanızı öneririm.

Veri Manipülasyonu:
--------------------
- df.apply(func)                  : Belirli bir fonksiyonu DataFrame'e uygulama
- df.drop_duplicates()            : Tekrarlayan satırları silme
- df.sample(n)                    : Rastgele n satırı örnekleme
- df.pivot_table(values, index, columns) : Pivot tablosu oluşturma

Veri Görselleştirme:
---------------------
- df.plot()                       : DataFrame verilerini görselleştirme
- df.plot(kind='bar')             : Çubuk grafik oluşturma
- df.plot(kind='hist')            : Histogram oluşturma
- df.plot(kind='scatter', x, y)   : Dağılım grafiği oluşturma

Veri Gruplama ve Agregasyon:
----------------------------
- df.groupby('sütun').sum()       : Sütuna göre gruplama ve toplam hesaplama
- df.groupby(['sütun1', 'sütun2']).mean() : Birden çok sütuna göre gruplama ve ortalama hesaplama

Veri Dönüşümleri:
------------------
- df.astype(dtype)                : Sütunların veri tiplerini dönüştürme
- df.replace(old, new)            : Değerleri değiştirme
- pd.to_datetime('sütun')         : Tarih/saat sütununu datetime veri tipine dönüştürme

Veri İndeksleme ve Seçme:
--------------------------
- df.loc[row_index, column_index] : Belirli satır ve sütunları seçme (etiket bazlı indeksleme)
- df.iloc[row_index, column_index] : Belirli satır ve sütunları seçme (konum bazlı indeksleme)
- df.loc[df['sütun'] > değer]     : Koşula göre satırları seçme (etiket bazlı indeksleme)
- df.iloc[boolean_condition]       : Koşula göre satırları seçme (konum bazlı indeksleme)

Bu cheatsheet, Pandas'ın temel işlemlerini kapsayan bir genel bakış sunmaktadır. Pandas'ın daha fazla işlevselliğini ve ayrıntısını keşfetmek için dokümantasyona başvurmanızı öneririm.

"""

"""
---------------------------------------------------
|              Seaborn Cheatsheet                  |
---------------------------------------------------

Seaborn, veri görselleştirmesi için kullanılan Python kütüphanelerinden biridir. İşte Seaborn kütüphanesiyle sık kullanılan bazı işlevler:

Veri Görselleştirme:
---------------------
- import seaborn as sns             : Seaborn kütüphanesini içe aktarma
- sns.set_style('style')            : Grafiklerin görünüm stilini ayarlama (darkgrid, whitegrid, dark, white, ticks)
- sns.set_palette('palette')        : Grafiklerde kullanılacak renk paletini ayarlama (deep, muted, bright, pastel, dark, colorblind)
- sns.countplot(x='sütun', data=df)  : Sütuna göre kategorik sayımların çubuk grafiğini oluşturma
- sns.barplot(x='sütun1', y='sütun2', data=df) : İki değişken arasındaki ilişkiyi çubuk grafiği ile gösterme
- sns.scatterplot(x='sütun1', y='sütun2', data=df) : İki değişken arasındaki ilişkiyi nokta grafiği ile gösterme
- sns.lineplot(x='sütun1', y='sütun2', data=df)   : İki değişken arasındaki ilişkiyi çizgi grafiği ile gösterme
- sns.heatmap(data=df)              : Veri setinin ısı haritasını oluşturma
- sns.boxplot(x='sütun1', y='sütun2', data=df)    : Bir kategorik değişkenle bir sayısal değişken arasındaki ilişkiyi kutu grafiği ile gösterme
- sns.histplot(x='sütun', data=df, bins=10)       : Bir sütunun histogramını oluşturma
- sns.pairplot(data=df)              : Değişkenler arasındaki ilişkileri çiftli grafiklerle gösterme

Grafik Ayarları:
-----------------
- plt.title('Başlık')               : Grafik başlığını ayarlama
- plt.xlabel('X etiketi')           : X ekseninin etiketini ayarlama
- plt.ylabel('Y etiketi')           : Y ekseninin etiketini ayarlama
- plt.xticks(rotation=45)           : X eksenindeki etiketleri döndürme
- plt.legend()                      : Grafikteki açıklamaları gösterme
- plt.savefig('dosya.png')          : Grafikleri dosyaya kaydetme

İstatistiksel İlişkiler:
------------------------
- sns.regplot(x='sütun1', y='sütun2', data=df)    : İki değişken arasındaki lineer regresyon ilişkisini gösterme
- sns.lmplot(x='sütun1', y='sütun2', data=df, hue='sütun3') : Değişkenler arasındaki regresyon ilişkisini renklendirme
- sns.jointplot(x='sütun1', y='sütun2', data=df) : İki değişken arasındaki ilişkiyi ortaklık grafiğiyle gösterme
- sns.residplot(x='sütun1', y='sütun2', data=df) : Regresyon hatalarının dağılımını gösterme

Grafik Özelleştirmeleri:
-------------------------
- sns.set(rc={'figure.figsize':(width, height)}) : Grafik boyutunu ayarlama
- sns.set_context('context')        : Grafiklerin öğelerinin boyutunu ayarlama (paper, notebook, talk, poster)
- sns.set(font_scale=scale)         : Grafik yazı tipi ölçeğini ayarlama
- sns.set_palette(['color1', 'color2'])          : Özel bir renk paleti belirleme

Bu cheatsheet, Seaborn kütüphanesini kullanarak veri görselleştirmesi yaparken sık kullanılan işlevleri içermektedir. Seaborn'ın daha fazla işlevselliğini ve ayrıntısını keşfetmek için resmi Seaborn dokümantasyonuna başvurmanızı öneririm.

"""