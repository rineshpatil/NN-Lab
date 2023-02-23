#Author:kkb Date: 28 January 2023
import numpy as np

# definitions of activation functions

def mp(net, th):  # for MP neuron
    return 1 if net >= th else 0

def relu(net):
    return net if net > 0 else 0


def tlu(net):  # bipolar binary (discrete perceptron)
    return 1 if net > 0 else -1


def step(net):  # unipolar binary (discrete perceptron)
    return 1 if net > 0 else 0


def sigmoid(net, lm=1):  # Unipolar continuous perceptron
    return (1 / (1 + np.exp(-lm * net)))


def tanh(net, lm=1):  # bipolar continuous perceptron
    return 2 / (1 + np.exp(-lm * net)) - 1


class Neuron:
    def __init__(self,w):
        self.x = []
        self.w = w

    def net(self):
        nt = np.dot(self.x, self.w)
        return nt

class MPneuron(Neuron):
    ''' CLass for McCulloch Pit's Model of Neuron (uses mp activation function)'''
    def __init__(self,w,th):
        Neuron.__init__(self,w)
        self.th=th

    def out(self):
        nt=self.net()
        return mp(nt,self.th)

class dbPtron(Neuron):
    ''' CLass for Discrete Bipolar Perceptron (uses TLU activation function)'''

    def __init__(self, w):
        Neuron.__init__(self, w)
        self.c = 1 #learning constant

    def out(self):
        nt = self.net()
        return tlu(nt)

class duPtron(Neuron):
    ''' CLass for Discrete Unipolar Perceptron (uses TLU activation function)'''

    def __init__(self, w):
        Neuron.__init__(self, w)
        self.c = 1 #learning constant

    def out(self):
        nt = self.net()
        return step(nt)