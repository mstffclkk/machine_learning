import animals

"""mammal = animals.Mammal('regular mammal')
mammal.show_species()
mammal.make_sound()

dog = animals.Dog()
dog.show_species()
dog.make_sound()

cat = animals.Cat()
cat.show_species()
cat.make_sound()
"""

def main():
    mammal = animals.Mammal('regular mammal')
    dog = animals.Dog()
    cat = animals.Cat()

    print("Here are some animals and")
    print("the sounds they make.")

    show_mammal_info(mammal)
    print()
    show_mammal_info(dog)
    print()
    show_mammal_info(cat)
    print()
    show_mammal_info("I am a string")

def show_mammal_info(creature):
    if isinstance(creature, animals.Mammal):
        creature.show_species()
        creature.make_sound()
    else:
        print("That is not a Mammal!")
   

main()

