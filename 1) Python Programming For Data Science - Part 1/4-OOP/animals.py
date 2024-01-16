class Mammal:
    def __init__(self, speices):
        self.__speices = speices

    def show_species(self):
        print("I am a", self.__speices)

    def make_sound(self):
        print("Grrrrr")

class Dog(Mammal):
    def __init__(self):
        Mammal.__init__(self, 'Dog')

    def make_sound(self):
        print("Woof! Woof!")
class Cat(Mammal):
    def __init__(self):
        Mammal.__init__(self, 'Cat')

    def make_sound(self):
        print("Meow!")