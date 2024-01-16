# *args: fonksiyonlara istediğimiz kadar argüman göndermemizi sağlar
def my_func(*args):
    print(args)

my_func(1,2,3,4,5,6,7,8,9,10)


# **kwargs: fonksiyonlara istediğimiz kadar keyword argüman göndermemizi sağlar
def my_func(**kwargs):
    print(kwargs)

my_func(name = "Mert", surname = "Çobanov", age = 22)


# *args ve **kwargs aynı anda kullanılabilir
def my_func(*args, **kwargs):
    print(args)
    print(kwargs)

my_func(1,2,3,4,5,6,7,8,9,10, name = "Mert", surname = "Çobanov", age = 22)


# Lambda Fonksiyonu: tek satırda tanımlanabilen fonksiyonlardır
square = lambda num: num ** 2
result = square(5)


# Map Fonksiyonu: bir fonksiyon ve bir iterable obje alır ve her bir elemanı fonksiyona gönderir
def square(num):
    return num ** 2

numbers = [1,2,3,4,5,6,7,8,9,10]
result = list(map(square, numbers))

# map fonksiyonu ile lambda fonksiyonu kullanımı
numbers = [1,2,3,4,5,6,7,8,9,10]
result = list(map(lambda num: num ** 2, numbers))


# Filter Fonksiyonu: bir fonksiyon ve bir iterable obje alır ve her bir elemanı fonksiyona gönderir
def check_even(num):
    return num % 2 == 0

numbers = [1,2,3,4,5,6,7,8,9,10]
result = list(filter(check_even, numbers))


# Reduce Fonksiyonu
# Reduce fonksiyonu bir fonksiyon ve bir iterable obje alır ve her bir elemanı fonksiyona gönderir
# ve fonksiyonun döndürdüğü değerleri bir liste içerisinde döndürür

from functools import reduce

def sum(num1, num2):
    return num1 + num2

numbers = [1,2,3,4,5,6,7,8,9,10]
result = reduce(sum, numbers)









    