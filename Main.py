import sys
import getopt
import numpy as np
import ReinforcementLearning as rele


def location_helper(state_name, map_size_x, map_size_y):
    print(f"Enter the number of {state_name} state(s): ")
    win_state_count = int(input())
    map_location = []
    for i in range(win_state_count):
        print(f"Enter the coordinates of the {state_name} state {i+1} (x and y starting from 1)")
        x = int(input("X: ")) - 1
        y = int(input("Y: ")) - 1
        if x >= map_size_x or y >= map_size_y:
            print("Location is out of bound")
            sys.exit()
        map_location.append((x, y))

    return map_location


def main(argv):
    opts, args = getopt.getopt(argv, "hmrd:wvai:e:", [
                               "map", "reward", "weights", "discount", "verbose", "asynchronuous", "iterations", "epsilon"])

    verbose = False
    asynchronuous = False

    reward_normal = -0.02
    reward_win = 1
    reward_lose = -1

    discount = 0.99

    weight_forward = 0.8
    weight_left = 0.1
    weight_right = 0.1

    map_size_x = 4
    map_size_y = 3

    map_location_win = [(0, 3)]
    map_location_lose = [(1, 3)]
    map_location_block = [(1, 1)]

    iterations = 1000

    epsilon = 0.001

    help_string = 'Main.py -h -m -r -d [value] -w -v -a -i [value] -e'

    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-d", "--discount"):
            try:
                discount = float(arg)
            except ValueError:
                print("Discount must be a float")
                print(help_string)
                sys.exit()
        elif opt in ("-a", "--asynchronuous"):
            asynchronuous = True
        elif opt in ("-r", "--reward"):
            print("----- REWARD ASSISTANT -----")
            try:
                reward_normal = float(
                    input("Enter the reward for the normal state: "))
                reward_win = float(
                    input("Enter the reward for the win state: "))
                reward_lose = float(
                    input("Enter the reward for the lose state: "))
            except ValueError:
                print("Reward must be a float")
                print(help_string)
                sys.exit()
        elif opt in ("-w", "--weights"):
            print("----- WEIGHT ASSISTANT -----")
            try:
                weight_forward = float(
                    input("Enter the weight for the forward action: "))
                weight_left = float(
                    input("Enter the weight for the left action: "))
                weight_right = float(
                    input("Enter the weight for the right action: "))
            except ValueError:
                print("Weight must be a float")
                print(help_string)
                sys.exit()
        elif opt in ("-m", "--map"):
            print("----- MAP ASSISTANT -----")
            map_location_win = []
            map_location_lose = []
            map_location_block = []
            try:
                map_size_x = int(input("Enter the X size of the map: "))
                map_size_y = int(input("Enter the Y size of the map: "))
                map_location_win = location_helper(
                    "win", map_size_x, map_size_y)
                map_location_lose = location_helper(
                    "lose", map_size_x, map_size_y)
                map_location_block = location_helper(
                    "block", map_size_x, map_size_y)

            except ValueError:
                print("Input must be an integer")
                print(help_string)
                sys.exit()
        elif opt in ("-i", "--iterations"):
            try:
                iterations = int(arg)
            except ValueError:
                print("Iterations must be an integer")
                print(help_string)
                sys.exit()
        elif opt in ("-e", "--epsilon"):
            try:
                epsilon = float(arg)
            except ValueError:
                print("Epsilon must be a float")
                print(help_string)
                sys.exit()

    Reward = np.full((map_size_y, map_size_x), reward_normal)

    for location in map_location_win:
        Reward[location[0], location[1]] = reward_win

    for location in map_location_lose:
        Reward[location[0], location[1]] = reward_lose

    for location in map_location_block:
        Reward[location[0], location[1]] = 0

    print("Map: ")
    print(Reward)

    rl = rele.ReinforcementLearning(
        Reward, discount, weight_forward, weight_left, weight_right, map_location_win, map_location_lose, map_location_block)

    rl.do_reinforcement_learning(iterations, epsilon, asynchronuous, verbose)


if __name__ == "__main__":
    main(sys.argv[1:])
