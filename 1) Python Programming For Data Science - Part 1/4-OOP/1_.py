# class

class VeriBilimci():
    print("Bu bir sınıftır.")

# class attributes

class VeriBilimci():
    bolum = ''
    sql = 'Evet'
    deneyim_yili = 0
    bildigi_diller = []

# siniflarin ozelliklerine erişmek.
VeriBilimci.sql
VeriBilimci.bildigi_diller

# siniflarin ozelliklerini degistirmek.
VeriBilimci.sql = 'Hayir'

# sinif ornegi (instance)

ali = VeriBilimci()
ali.sql
ali.deneyim_yili
ali.bolum
ali.bildigi_diller.append('Python')


veli = VeriBilimci()
veli.sql
veli.deneyim_yili
veli.bolum
veli.bildigi_diller # veli'nin bildigi dilleri ali'nin bildigi dilleri ile ayni.!!!!!!

# ornek ozellikleri
class VeriBilimci():
    bolum = ''
    def __init__(self):
        self.bildigi_diller = []
        self.sql = ''
        self.deneyim_yili = 0
        print("init fonksiyonu cagirildi.")


VeriBilimci().bildigi_diller
VeriBilimci().bolum
VeriBilimci().sql
VeriBilimci().deneyim_yili

ali = VeriBilimci()
ali.bildigi_diller = ['Python', 'R']         # 1. yol
ali.bildigi_diller.append('Python')     # 2. yol
ali.bolum = 'istatistik' 
ali.sql = 'Evet'
ali.deneyim_yili = 1


veli = VeriBilimci()
veli.bildigi_diller = ['SAS', 'SPSS']
# veli.bildigi_diller.append(['R', 'Python']) # bu kısmı anlamadım.
veli.bolum = 'end_muh'
veli.sql = 'Hayir'
veli.deneyim_yili = 0


# ornek metodlari
class VeriBilimci(): 
    calisanlar = []                             # class attributes (sınıf nitelikleri)
    def __init__(self):                         # constructor (yapıcı metod)
        self.bildigi_diller = []                
        self.bolum = ''                         
    def dil_ekle(self, yeni_dil):               # instance method (örnek metodu)
        self.bildigi_diller.append(yeni_dil)   

ali = VeriBilimci()                             # ali bir VeriBilimci objesi
ali.bildigi_diller 
ali.bolum
ali.dil_ekle('R') 

veli = VeriBilimci()
veli.bildigi_diller
veli.dil_ekle('Python')


class Araba:
    def __init__(self, marka, model, hiz=0):
        self.marka = marka
        self.model = model
        self.hiz = hiz
    
    def hizlan(self, artis):
        self.hiz += artis
    
    def yavaşla(self, azalis):
        self.hiz -= azalis
    
    def dur(self):
        self.hiz = 0
    
    def __str__(self):
        return f"{self.marka} {self.model} - Hız: {self.hiz} km/saat"


# Araba örnekleri oluşturma
araba1 = Araba("Ford", "Mustang")
araba2 = Araba("Tesla", "Model S", 120)

# Arabaların durumlarını gösterme
print(araba1)  # Ford Mustang - Hız: 0 km/saat
print(araba2)  # Tesla Model S - Hız: 120 km/saat

# Arabaların hızlanması ve yavaşlaması
araba1.hizlan(50)
araba2.yavaşla(30)

# Arabaların güncel hızlarını gösterme
print(araba1)  # Ford Mustang - Hız: 50 km/saat
print(araba2)  # Tesla Model S - Hız: 90 km/saat

# Bir arabanın durması
araba1.dur()

# Arabanın güncel hızını gösterme
print(araba1)  # Ford Mustang - Hız: 0 km/saat
