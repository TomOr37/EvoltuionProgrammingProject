from eckity.genetic_operators.genetic_operator import GeneticOperator
import math, random
import numpy as np


class ModelParamSwapCrossOver(GeneticOperator):
    def __init__(self, probability=0.1, arity=2, events=None):
        super().__init__(probability, arity, events)

    def apply(self, payload):
        """
        Swap some of the parameters between 2 networks:
        in probability of 0.5 ,  choosing which parameter to choose( from which network).

        Then, after creating new weights and biases, in probability of 0.5 choose which net to modify.

        Parameters
        -------
        payload: object
            relevant data for the applied operator (usually a list of individuals).


        :returns

            Updated payload after crossover.
        """
        first_individual = payload[0]
        second_individual = payload[1]
        first_weigths = first_individual.model.get_weigths()
        first_bias = first_individual.model.get_bias()
        second_weigths = second_individual.model.get_weigths()
        second_bias = second_individual.model.get_bias()

        new_weigths = []
        new_bias = []

        for weight1, weight2 in zip(first_weigths, second_weigths):
            choosing_probability = np.random.uniform(0, 1) > 0.5

            if choosing_probability:

                new_weigths.append(weight1)

            else:
                new_weigths.append(weight2)

        for bias1, bias2 in zip(first_bias, second_bias):
            choosing_probability = np.random.uniform(0, 1) > 0.5

            if choosing_probability:

                new_bias.append(bias1)

            else:
                new_bias.append(bias2)

        choosing_probability = np.random.uniform(0, 1) > 0.5

        if choosing_probability:

            first_individual.model.init_linear(new_weigths, new_bias)

        else:

            second_individual.model.init_linear(new_weigths, new_bias)

        return payload


class ModelAddDistMutation(GeneticOperator):
    def __init__(self, probability=0.2, arity=1, weight_size=3, bias_size=1, events=None):
        super().__init__(probability, arity, events)
        self.weight_size = weight_size
        self.bias_size = bias_size

    def apply(self, payload):
        """
        Apply the operator, on payload.

        This mutation operator adds with probabilty of 0.2 random value sampled from the same initialization distribution of
        the neural net to the current weigths and biases

        Parameters:
        -------
        payload: object
            relevant data for the applied operator (usually a list of individuals)
        """

        for individual in payload:
            additional_weights, additional_bias = self.distrubtion_sampler()
            old_weigths = individual.model.get_weigths()
            old_bias = individual.model.get_bias()
            individual.model.init_linear(old_weigths + additional_weights, old_bias + additional_bias)

        return payload

    def distrubtion_sampler(self):
        """
        default init for linear layer (pytorch implementation):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


        :returns
            new initialized weights and biases.
        """
        stdv = 1. / math.sqrt(self.weight_size)
        weights_dist = np.random.uniform(low=-stdv, high=stdv, size=self.weight_size)
        bias_dist = np.random.uniform(low=-stdv, high=stdv, size=self.bias_size)
        return weights_dist, bias_dist
