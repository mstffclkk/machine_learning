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

# miras yapıları (inheritance)

class Employees():
    def __init__(self, FirstName, LastName, Address):
        self.FirstName = FirstName
        self.LastName = LastName
        self.Address = Address

class DataScience(Employees):
    def __init__(self, Programming):
        self.Programming = Programming

class Marketing(Employees):
    def __init__(self, StoryTelling):
        self.StoryTelling = StoryTelling

dataScience = DataScience('Python')
dataScience.FirstName = 'Ali'
dataScience.LastName = 'Yilmaz'
dataScience.Address = 'İstanbul'
print(dataScience.FirstName)
