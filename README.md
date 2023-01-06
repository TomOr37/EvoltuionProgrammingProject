# Evolution AI for Flappy bird
### Project in "Topics in appliction of computer science".
### by: Tom or, Noa Cohen , Yonatan Hod


## Intoduction

"Flappy bird" is an arcade game popularized in the 2010's built for mobile phones.
#### Gameplay:
The player play as the bird, while the goal of the game is to dodge some pipes 
by "jumping" and getting far as one can. The pipes position varies in the game. The score increase by 1 every time the bird 
successfully passes through a pipe.

![Alt text](readme_images/gameplay.png "Game play")
###### Figure 1. Gameplay. When the bird touches a pipe, the player loses ,and the game is over.

## Problem Description
We wanted to develop an algorithm which will play the game. That is, giving
the game settings (pipes position, bird position), the algorithm will decide when to "jump"
and getting as far as can. <br />
Developing an algorithm using ordinary methods (i.e not learning) can be quite difficult:
defining the rule when to jump may require a large amount of trial and error and maybe some knowledge of the
game physics.<br />
Therefore, we show to power of genetic programming which will learn to play the game.<br />

More specifically, we will use a simple neural network to decide when to jump. Usually,
neural networks are optimized through back-propagation, but for our problem setting
it is not suitable, as we don't have "train data". We will demonstrate
how we can "optimize" it through GP (that is, finding good parameters for our network).

Overall, the problem is to search a set  `P` of parameters for a specific neural network architecture `N`,
so `N` given the settings of `P` will allow the bird to get as far as possibly.

## Solution through GP

For the usage of GP we need to represent an individual,
fitness method and genetic operators.

### Individual
Our individual is the bird, with the property of a neural network `model`.
### Fitness
We evaluate a fitness of an individual by simulating the game with that individual: <br />
For every 30ms that the individual still has not lost, we increased the fitness by 0.05. <br />
For every pair of pipes that the individual passed,we increased the fitness by 1.5 .<br />
#### Notes:
If the individual left the frame of the game (jump too many times, or didn't jump at all) we count it as loss.
We added a "penalty" of 1 to the fitness value when the individual lost. we saw it gave better results. <br>
The individual decides to jump or not every 30ms.

### `model`
`model` is a simple feed-forward neural network with 3 input nodes, and 1 output node - a total of 3 weights and 1 bias parameters to find.
We used tanh activation function (so the result is between [-1,1]) and if `model` output was higher than 0, the bird jumped.<br/>
We fed the model the (current) height of the individual,the distance from the top pipe, and the distance from the bottom pipe.

### Encoding
We encode an individual through the parameters of his `model` .

### Population
The initialized population is a group of birds, each one has random parameters for `model` sampled
from distribution `D`.

### Genetic Operators
We defined 2 genetic operates: `ModelAddDistMutation` , `ModelParamSwapCrossOver`.

#### `ModelAddDistMutation`:
`ModelAddDistMutation` is a mutation operator. It takes the encoding of an individual, i.e the parameters
of `model`, and adds to each parameter random value sampled from `D`.

#### `ModelParamSwapCrossOver`:
`ModelAddDistMutation` is a cross-over operator. It takes the encodings of  2 individuals: `A`,`B`, and create new encoding `C` the following way:<br/>
For each new parameter `k` , with probability of `p1`, take the corresponding parameter `k'` of `A` and set `k=k'`, otherwise, take from `B` (we set `p1`=0.5).
<br/>.Then, with probability of `p2` swap `A` with `C`, otherwise swap with `B` (we set `p2`=0.5).

### Selection method
We used Tournament Selection of size 3.


## Software Overview:

+ `pygame` package was used to implement the game logic (collision,drawing etc...).
+ `eckity` for implementing GP logic.
+ `pytorch` for the neural network logic.

### Bird

`Bird` is a class which represents individual. Therefore, it is  inheriting
from `eckity.individual.Individual` class. <br>Main Methods: <br />
+ `jump`: jumping behavior
+ `move`: moving behavior <br>
+ `get_mask`: return the masking of the bird. Used to check collision
+ `draw`: draw the bird.
+ `excute`: this method is called at the end of the program on the best individual. its show a simulation of the individual playing.

#### Notes
The images needed to display the bird are not a property of the `Bird` class. This is 
because there is a serialization problem when they are.

### Pipe, Base 
Pipe and Base are 2 classes which represent the pipes, and the base (the floor and the background) of 
the game.
The Pipe class implements  `collision` method. this method take as a parameter a Bird instance, and checks
if collision happened.

#### Notes:
Pipe and Base implements `move`, so in the screen they will show as moving.

### BirdEvaluator
Evaluator class for individual.( inheriting from `eckity.evaluators.simple_individual_evaluator.SimpleIndividualEvaluator`)
The class main method is `_evaluate_individual` which evaluates an individual:
simulates a game given the individual.

#### Notes:
`_evaluate_individual` uses a helper method `eval` which preform the simulation logic.
`eval` takes as flag `show_game` which indicates whether to display the gameplay.
In this context it is passed as `False` , to help faster computation.
`eval` also takes `limit` parameter which terminates the game if the individual passed
with his fitness `limit` value.


### FFmodel
This class implements `model` property of individual. It's inheriting from `torch.nn.Module`.
An instance of the class is `model` neural network described above.

#### Note

### BirdCreator
This class implements how individuals are made (inheriting from `eckity.creators.creator.Creator`). The class main method is `_create_individuals`.
We create `n` individual by creating `n` `Bird` instance with random
