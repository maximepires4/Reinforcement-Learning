# Reinforcement Learning

import numpy as np
import matplotlib.pyplot as plt

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3


class ReinforcementLearning:

    def __init__(self, reward, discount, forward_weight, left_weight, right_weight, map_location_win, map_location_lose, map_location_block):
        self.reward = reward
        self.discount = discount
        self.weights = {
            NORTH: (forward_weight, 0, left_weight, right_weight),
            SOUTH: (0, forward_weight, right_weight, left_weight),
            WEST: (left_weight, right_weight, forward_weight, 0),
            EAST: (right_weight, left_weight, 0, forward_weight)
        }
        self.map_location_win = map_location_win
        self.map_location_lose = map_location_lose
        self.map_location_block = map_location_block

    def check_ending_state(self, x, y):
        return (x, y) in self.map_location_win or (x, y) in self.map_location_lose

    def get_next_state_value(self, x, y, action, V):

        north_weight, south_weight, west_weight, east_weight = self.weights[action]

        if x - 1 >= 0 and (x - 1, y) not in self.map_location_block:
            north_value = north_weight * V[(x - 1), y]
        else:
            north_value = north_weight * V[x, y]

        if x + 1 < V.shape[0] and (x + 1, y) not in self.map_location_block:
            south_value = south_weight * V[(x + 1), y]
        else:
            south_value = south_weight * V[x, y]

        if y - 1 >= 0 and (x, y - 1) not in self.map_location_block:
            west_value = west_weight * V[x, (y - 1)]
        else:
            west_value = west_weight * V[x, y]

        if y + 1 < V.shape[1] and (x, y + 1) not in self.map_location_block:
            east_value = east_weight * V[x, (y + 1)]
        else:
            east_value = east_weight * V[x, y]

        return north_value + south_value + west_value + east_value

    def evaluate_state_value(self, x, y, V):
        if self.check_ending_state(x, y) or (x, y) in self.map_location_block:
            return self.reward[x, y]

        max = float('-inf')
        for action in range(0, 4):
            value = self.get_next_state_value(x, y, action, V)
            if value > max:
                max = value

        return self.reward[x, y] + self.discount * max

    def evaluate_policy(self, V):
        P = np.zeros(V.shape)
        for i in range(0, V.shape[0]):
            for j in range(0, V.shape[1]):
                if (i, j) in self.map_location_win:
                    P[i, j] = -1
                elif (i, j) in self.map_location_lose:
                    P[i, j] = -2
                elif (i, j) in self.map_location_block:
                    P[i, j] = -3
                else:
                    max = float('-inf')
                    for action in range(0, 4):
                        value = self.get_next_state_value(i, j, action, V)
                        if value > max:
                            max = value
                            P[i, j] = action

        return P

    def do_reinforcement_learning(self, max_iterations, epsilon, asynchronuous=False, verbose=False):
        if verbose:
            error_list = []

        Value_function = np.zeros(self.reward.shape)
        deltaV = np.full(Value_function.shape, 1)

        for iter in range(max_iterations):
            V_old = np.array(Value_function)

            if asynchronuous:
                V_new = np.array(Value_function)
            else:
                V_new = Value_function

            for i in range(Value_function.shape[0]):
                for j in range(Value_function.shape[1]):
                    V_new[i, j] = self.evaluate_state_value(
                        i, j, Value_function)

            if asynchronuous:
                Value_function = V_new

            deltaV = np.abs(V_new - V_old)
            maxV = np.max(deltaV)
            if verbose:
                error_list.append(maxV)

            if maxV <= epsilon:
                if verbose:
                    print(f"Algorithms converged after {iter+1} iterations with final delta error of {maxV}:")
                break

            if iter == max_iterations - 1 and verbose:
                print(
                    f"Algorithms did not converged after {iter+1} iterations with final delta error of {maxV}:")
                
        if verbose:
            print(Value_function)

        Policy = self.evaluate_policy(Value_function)

        print("Final policy (N - North, S - South, W - West, E - East, T - target, L - Lose, X - Wall):")
        Policy_formatted = np.full(Policy.shape, 'X')
        for i in range(Value_function.shape[0]):
            for j in range(Value_function.shape[1]):
                if Policy[i, j] == 0:
                    Policy_formatted[i, j] = 'N'
                elif Policy[i, j] == 1:
                    Policy_formatted[i, j] = 'S'
                elif Policy[i, j] == 2:
                    Policy_formatted[i, j] = 'W'
                elif Policy[i, j] == 3:
                    Policy_formatted[i, j] = 'E'
                elif Policy[i, j] == -1:
                    Policy_formatted[i, j] = 'T'
                elif Policy[i, j] == -2:
                    Policy_formatted[i, j] = 'L'
                elif Policy[i, j] == -3:
                    Policy_formatted[i, j] = 'X'
        print(Policy_formatted)

        if verbose:
            plt.figure(5)
            plt.plot(error_list)
            plt.xlabel('Iterations')
            plt.ylabel('Error')
            plt.show()
