# automobile superclass

class Automobile:
    def __init__(self, make, model, milage, price):
        self.__make = make
        self.__model = model
        self.__milage = milage
        self.__price = price

    def set_make(self, make):
        self.__make = make 
    
    def set_model(self, model):
        self.__model = model
    
    def set_milage(self, milage):
        self.__milage = milage

    def set_price(self, price):
        self.__price = price
    
    def get_make(self):
        return self.__make

    def get_model(self):
        return self.__model
    
    def get_milage(self):
        return self.__milage
    
    def get_price(self):
        return self.__price
    
    def __str__(self):
        return f"Make: {self.__make}\nModel: {self.__model}\nMilage: {self.__milage}\nPrice: {self.__price}"
    
# car subclass

class Car(Automobile):
    def __init__(self, make, model, milage, price, doors):
        Automobile.__init__(self, make, model, milage, price)
        self.__doors = doors

    def set_doors(self, doors):
        self.__doors = doors
    
    def get_doors(self):
        return self.__doors
    
    def __str__(self):
        return f"Make: {self.get_make()}\nModel: {self.get_model()}\nMilage: {self.get_milage()}\nPrice: {self.get_price()}\nDoors: {self.__doors}"
    
# truck subclass

class Truck(Automobile):
    def __init__(self, make, model, milage, price, drive_type):
        Automobile.__init__(self, make, model, milage, price)
        self.__drive_type = drive_type

    def set_drive_type(self, drive_type):
        self.__drive_type = drive_type
    
    def get_drive_type(self):
        return self.__drive_type
    
    def __str__(self):
        return f"Make: {self.get_make()}\nModel: {self.get_model()}\nMilage: {self.get_milage()}\nPrice: {self.get_price()}\nDrive Type: {self.__drive_type}"

# suv subclass

class SUV(Automobile):
    def __init__(self, make, model, milage, price, pass_cap):
        Automobile.__init__(self, make, model, milage, price)
        self.__pass_cap = pass_cap

    def set_pass_cap(self, pass_cap):
        self.__pass_cap = pass_cap
    
    def get_pass_cap(self):
        return self.__pass_cap
    
    def __str__(self):
        return f"Make: {self.get_make()}\nModel: {self.get_model()}\nMilage: {self.get_milage()}\nPrice: {self.get_price()}\nPassenger Capacity: {self.__pass_cap}"

# driver









