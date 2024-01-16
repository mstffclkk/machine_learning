import random

class Coin:
    def __init__(self):
        self.sideup = 'Heads'
    def toss(self):
        if random.randint(0,1) == 0:
            self.sideup = 'Heads'
        else:
            self.sideup = 'Tails'
    def get_sideup(self):
        return self.sideup
    
my_coin = Coin()
def flip(coin):
    coin.toss()
    return coin.get_sideup()

coin_list = list(map(flip, [my_coin]*10))


