from torch import nn, tensor
import numpy


"""
default init for linear layer:
stdv = 1. / math.sqrt(self.weight.size(1))
self.weight.data.uniform_(-stdv, stdv)
if self.bias is not None:
    self.bias.data.uniform_(-stdv, stdv)
"""



class FFModel(nn.Module):  ##currently default ininiliazing of torch
    def __init__(self):
        # Linear function
        super(FFModel, self).__init__()

        self.fc1 = nn.Linear(3, 1)  # 3 input features

        # Non-linearity
        self.tanh = nn.Tanh()  # activion function

    def forward(self, x):
        """

        :param x: vector of shape (1,3) - individual y pos, and the 2 pipes height
        :return: number between -1 1, certainty of jumping
        """
        x = tensor(x,dtype=float)

        out = self.fc1(x)


        out = self.tanh(out)

        return out  # number between -1 and 1

    def get_weigths(self):
        return self.fc1.weight.detach().numpy()

    def get_bias(self):
        return self.fc1.bias.detach().numpy()

    def init_linear(self, new_weigth, new_bias, random_distrubtion=None):
        self.fc1.weight.data = tensor(new_weigth, requires_grad=False)
        self.fc1.bias.data = tensor(new_bias, requires_grad=False)


