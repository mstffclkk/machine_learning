
###############################################
# KOŞULLAR (CONDITIONS)
###############################################

# True-False'u hatırlayalım.
1 == 1
1 == 2

# if
if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11

if number == 10:
    print("number is 10")

number = 10
number = 20


def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)

#######################
# else
#######################

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)


def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

#######################
# elif
#######################

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

number_check(6)
