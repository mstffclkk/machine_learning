from vehicles import Car , Truck , SUV

def main():
    car = Car('Honda', 2001 , 15000, 12000, 2)
    truck = Truck('Ford', 2002 , 12000, 20000, '4WD')
    suv = SUV('Jeep', 2000, 18000, 25000, 5)

    print(car)
    print()
    print(truck)
    print()
    print(suv)

main()
