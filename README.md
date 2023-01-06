#Evolution AI for Flappy bird
###Project in "Topics in appliction of computer science".
###by: Tom or, Noa Cohen , Yonatan Hod


##Intoduction

"Flappy bird" is an arcade game popularized in the 2010's built for mobile phones.
####Gameplay:
The player play as the bird, while the goal of the game is to dodge some pipes 
by "jumping" and getting far as one can. The pipes position varies in the game. The score increase by 1 every time the bird 
successfully passes through a pipe.
####Figuare 1. Gameplay
![Alt text](readme_images/gameplay.png "Game play")
######Gameplay. When the bird touches a pipe, the player lose and the game is over.

##Problem Description
We wanted to develop an algorithm which will play the game. That is, giving
the game settings (pipes position, bird position), the algorithm will decide when to "jump"
and getting as far as can. <br />
Developing an algorithm using ordinary methods (i.e not learning) can be quite difficult:
defining the rule when to jump may require a large amount of trial and error and maybe some knowledge of the
game physics.<br />
Therefore, we show to power of genetic programming which will learn to play the game.<br />

More specifically, we will use a simple nerual network to decide when to jump. Usually,
neural networks are optimized through back-propagation, but for our problem setting
it is not suitable, as we don't have "train data". We will demonstrate
how we can "optimize" it through GP (that is, finding good parameters for our network).

Overall, the problem is to search a set  `P` of parameters for a specific neural network architecture `N`,
so `N` given the settings of `P` will allow the bird to get as far as possibly.

##Solution through GP

For the usage of GP we need to represent an individual,
fitness method and genetic operators.

###Individual
Our individual is the bird, with the property of a neural network `model`.
###Fitness
We evaluate a fitness of an individual by simulating the game with that individual: <br />
For every 30ms that the individual still has not lost, we increased the fitness by 0.05. <br />
For every pair of pipes that the individual passed,we increased the fitness by 1.5 .<br />
####Notes:
If the individual left the frame of the game (jump to many times, or didn't jump at all) we count it as loss.
We added a "penalty" of 1 to the fitness value when the individual lost. we saw it gave better results. <br>
The individual decides to jump or not every 30ms.

###`model`
`model` is a simple feed-forward neural network with 3 input nodes, and 1 output node - a total of 3 weights and 1 bias parameters to find.
We used tanh activation function (so the result is between [-1,1]) and if `model` output was higher than 0, the bird jumped.<br/>
We fed the model the (current) height of the individual,the distance from the top pipe, and the distance from the bottom pipe.

###Encoding
We encode an individual through the parameters of his `model` .

###Genetic Operators
We defined 2 genetic operates: `ModelAddDistMutation` , `ModelParamSwapCrossOver`.






