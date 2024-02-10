
###############################################
# DÖNGÜLER (LOOPS)
###############################################
# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)


for salary in salaries:
    print(int(salary*20/100 + salary))

for salary in salaries:
    print(int(salary*30/100 + salary))

for salary in salaries:
    print(int(salary*50/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 20))


salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))



#######################
# Uygulama - Mülakat Sorusu
#######################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

a = "hi my name is john and i am learning python"

####### 1
def swap(string):
    b = ""
    for i in range(len(string)):
        if i % 2 == 0:
            b += string[i].upper()
        else:
            b += string[i].lower()
    print(b)
swap(a)

####### 2
def swap(string):
    b = ""
    for index, char in enumerate(string):
        if index % 2 == 0:
            b += char.upper()
        else:
            b += char.lower()
    print(b)
swap(a)


range(len("miuul"))
range(0, 5)

for i in range(len("miuul")):
    print(i)

# 4 % 2 == 0
# m = "miuul"
# m[0]

def alternating(string):
    new_string = ""
    # girilen string'in index'lerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir.
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir.
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("miuul")

#######################
# break & continue & while
#######################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)


for salary in salaries:
    if salary == 3000:
        continue
    print(salary)


# while

number = 1
while number < 5:
    print(number)
    number += 1

#######################
# Enumerate: Otomatik Counter/Indexer ile for loop
#######################

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)


#######################
# Uygulama - Mülakat Sorusu
#######################
# divide_students fonksiyonu yazınız.
# Çift indexte yer alan öğrencileri bir listeye alınız.
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]

######### 1
def divide_students(string):
    l1 = []
    l2 = []
    for i, c in enumerate(string):
        if i % 2 == 0:
            l1.append(c)
        else:
            l2.append(c)
    return l1 + l2
divide_students(students)

######### 2
def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)
st[0]
st[1]


#######################
# alternating fonksiyonunun enumerate ile yazılması
#######################

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")
         
#######################
# Zip
#######################

## 1 - List
students = ["John", "Mark", "Venessa", "Mariam"]
departments = ["mathematics", "statistics", "physics", "astronomy"]
ages = [23, 30, 26, 22]
list(zip(students, departments, ages))

a = [1, 2, 3]
b = [4, 5, 6]
list(zip(a, b))

for i, j in zip(a, b):
    print(i, j)

## 2 - Dict
d1 = {'a': 1, 'b': 2, 'c': 3}
d2 = {'d': 4, 'e': 5, 'f': 6}
dict(zip(d1, d2))


###############################################
# lambda, map, filter, reduce
###############################################

def summer(a, b):
    return a + b

summer(1, 3) * 9

new_sum = lambda a, b: a + b

new_sum(4, 5)

##### MAP : bir fonksiyonu bir listenin üzerinde uygulamak için kullanılır.
##### map(fonksiyon, iteratif nesne)
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x
new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

# for yerine map kullanabiliriz.
list(map(new_salary, salaries))

# lambda ile yazalım.
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2 , salaries))

##### FILTER - bir fonksiyonu bir listenin üzerinde uygulayarak True dönenleri bir liste içinde saklar.
##### sorgu fonksiyonudur.
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

##### REDUCE - bir fonksiyonu bir listenin üzerinde uygulayarak tek bir değer döndürür.
from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)
