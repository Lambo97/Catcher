from os import system


system("python ppo.py --discrete --actions {}".format(7))

for i in range(10):
        print("-------------------------------- NEW TEST {} DISCRETE {} -----------------------------------".format(i, 15))
        system("python ppo.py --discrete --actions {}".format(15))