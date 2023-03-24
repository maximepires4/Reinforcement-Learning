# Reinforcement learning on a robot on a map

Implements a simple version of reinforcement learning on a robot which progress into a map.
The map contains a target point, a lose point and a wall inside, which is all customizable.

The robot moves forward, to the left or to the right according to a given action and some probabilities to add some noise.
The probabilities are also customizable.

The algorithm output the policy of the robot (i.e which direction the robot should head, heading into a impossible direction will result into not moving).

## Usage

```
python Main.py -h -m -r -d [value] -w -v -a -i [value] -e
```

## Parameters

* Help

`-h --help` for printing the help

* Map

`-m --map` for customizing the map (default: X=4, Y=10, target=(0, 3), lose=(1, 3), wall=(1, 1))

* Reward

`-r --reward` for customizing the reward (default: normal=-0.02, win=1, lose=-1)

* Discount

`-d --discount [value]` for setting the discount factor (default: 0.99)

* Weights

`-w --weights` for launching the weights editor (default: forward=0.8, left=0.1, right=0.1)

* Verbose

`-v --verbose` for printing final values and plotting error graph (default: False)

* Asynchronuous

`-a --asynchronuous` for launching the asynchronuous version of the algorithm (default: False)

* Iterations

`-i --iterations [value]` for setting the number of iterations (default: 1000)

* Epsilon

`-e --epsilon` for setting the epsilon value for error (default: 0.001)