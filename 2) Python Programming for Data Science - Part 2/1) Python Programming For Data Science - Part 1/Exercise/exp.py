## Görev 2:  Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, 
## kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight."
text = text.upper().replace(",", " ").replace(".", " ").split()

## Görev 3:   Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D","A","T","A","S","C","I","E","N","C","E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
len(lst)
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
lst[0]
lst[10]
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
lst[0:4]
# Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
# Adım 5: Yeni bir eleman ekleyiniz.
lst.append("E")
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

## Görev 4:   Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {"Christian": ["America",18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.
dict.keys()
# Adım 2: Value'lara erişiniz.
dict.values()
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]              # 1
dict.update({"Ahmet": ["Turkey", 24]})      # 2
# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")

## Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
## return eden fonksiyon yazınız

l = [2,13,18,93,22]

def func(list):
    even = [i for i in list if i % 2 == 0]
    odd = [i for i in list if i % 2 != 0]

    return even, odd

even_l, odd_l = func(l)

## Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
## bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de 
## tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep","Ece"]

for index, student in enumerate(ogrenciler):
    if index < 3:
        print(f"Mühendislik fakültesinde {index+1}. sirada {student} var.")
    else:
        print(f"Tip fakültesinde {index-2}. sirada {student} var.")

## Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
## almaktadır. Zip kullanarak ders bilgilerini bastırınız.
ders_kodu = ["CMP1005", "PSY1001", "MAT1002", "STA1003", "STA1004", "CMP1006", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

for ders, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders} kodlu dersin kontenjanı {kontenjan} kişidir .")


## Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
## eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1 = set(["data", "python"])
kume2 = set(["data", "function","qcut","lambda","python", "miuul"])

def func(kume1, kume2):
    if kume1.issuperset(kume2):
        print(kume1.intersection(kume2))
    else:
        print(kume2.difference(kume1))

func(kume1, kume2)


