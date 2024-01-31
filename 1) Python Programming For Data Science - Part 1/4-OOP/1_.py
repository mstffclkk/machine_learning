# class

class Scientist():
    print("Bu bir siniftir.")

# class attributes

class Scientist():
    bolum = ''
    sql = 'Evet'
    deneyim_yili = 0
    bildigi_diller = []

# Accessing the properties of classes.
Scientist.sql
Scientist.bildigi_diller

# siniflarin ozelliklerini degistirmek.
Scientist.sql = 'Hayir'

# sinif ornegi (instance)

ali = Scientist()
ali.sql
ali.deneyim_yili
ali.bolum
ali.bildigi_diller.append('Python')


veli = Scientist()
veli.sql
veli.deneyim_yili
veli.bolum
veli.bildigi_diller # veli'nin bildigi dilleri ali'nin bildigi dilleri ile ayni.!!!!!!

# ornek ozellikleri
class Scientist():
    bolum = ''
    def __init__(self):
        self.bildigi_diller = []
        self.sql = ''
        self.deneyim_yili = 0
        print("init fonksiyonu cagirildi.")


Scientist().bildigi_diller
Scientist().bolum
Scientist().sql
Scientist().deneyim_yili

ali = Scientist()
ali.bildigi_diller = ['Python', 'R']         # 1. yol
ali.bildigi_diller.append('Python')     # 2. yol
ali.bolum = 'istatistik' 
ali.sql = 'Evet'
ali.deneyim_yili = 1


veli = Scientist()
veli.bildigi_diller = ['SAS', 'SPSS']
# veli.bildigi_diller.append(['R', 'Python']) # bu kısmı anlamadım.
veli.bolum = 'end_muh'
veli.sql = 'Hayir'
veli.deneyim_yili = 0


# ornek metodlari
class Scientist(): 
    calisanlar = []                             # class attributes (sınıf nitelikleri)
    def __init__(self):                         # constructor (yapıcı metod)
        self.bildigi_diller = []                
        self.bolum = ''                         
    def dil_ekle(self, yeni_dil):               # instance method (örnek metodu)
        self.bildigi_diller.append(yeni_dil)   

ali = Scientist()                             # ali bir VeriBilimci objesi
ali.bildigi_diller 
ali.bolum
ali.dil_ekle('R') 

veli = Scientist()
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
