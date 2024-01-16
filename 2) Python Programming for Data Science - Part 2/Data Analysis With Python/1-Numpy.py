###############################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###############################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

#############################################
# NUMPY
#############################################

# Neden NumPy? (Why Numpy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

#############################################
# Why Numpy?
#############################################
import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]
ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

# Numpy ile
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#############################################
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
#############################################
import numpy as np

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5])) 
np.zeros(10, dtype=int) 
np.random.randint(0, 10, size=10) # np.random.randint(start, end, size) -> end dahil değil
"""
# 0 ile 9 arasinda (dahil) rastgele bir tam sayi üretme
random_number = np.random.randint(10)
print(random_number)

# 1 ile 100 arasinda (hariç) rastgele bir tam sayi üretme
random_number = np.random.randint(1, 100)
print(random_number)

# 10 ile 20 arasinda (dahil) 5 adet rastgele tam sayi üretme
random_numbers = np.random.randint(10, 21, size=5)
print(random_numbers)

# 0 ile 9 arasinda (dahil) 2x3 boyutunda rastgele tam sayilar içeren bir dizi oluşturma
random_array = np.random.randint(10, size=(2, 3))
print(random_array)
"""
np.random.normal(10, 4, (3, 4)) # np.random.normal(mean, std, size)

#############################################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
#############################################
import numpy as np

# ndim: boyut sayisi
# shape: boyut bilgisi
# size: toplam eleman sayisi
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#############################################
# Yeniden Şekillendirme (Reshaping) 
# reshape: verinin boyutunu değiştirmek için kullanılır.
#############################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=(3, 3))           # 3x3 matris
np.random.randint(1, 10, size=9).reshape(3, 3)  # 3x3 matris

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

#############################################
# Index Seçimi (Index Selection)
#############################################
import numpy as np
# 1D array
a = np.random.randint(10, size=10)
a[0]
a[0:5]  # 0'dan 5'e kadar (5 dahil değil)
a[0] = 999

# 2D array
m = np.random.randint(10, size=(3, 5))

m[0, 0] #m[0][0] 
m[1, 1]
m[2, 3]

m[2, 3] = 999
m[2, 3] = 2.9

m[:, 0]
m[1, :]
m[0:2, 0:3]

#############################################
# Fancy Index
#############################################
import numpy as np

v = np.arange(0, 30, 3) # 0'dan 30'a kadar(hariç) 3'er 3'er artan sayılar
v[1]

catch = [1, 2, 3]
v[catch]    # v'nin 1, 2, 3. elemanları getirir.

#############################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#######################
# Klasik döngü ile
#######################
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#######################
# Numpy ile
#######################
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#############################################
# Matematiksel İşlemler (Mathematical Operations)
#############################################
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)

#######################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
#######################

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)