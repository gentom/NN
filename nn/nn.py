#!/usr/bin/env python
# coding: utf-8
import math
import random

# Work In Progress.

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def sigmoid(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def sigmoid_transfer(output):
    return output * (1.0 - output)

class NN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.network = list()
        self.hidden_layer = [{'weights': [random.random() for i in range(self.n_inputs + 1)]}
                        for i in range(self.n_hidden)]
        self.network.append(self.hidden_layer)
        self.output_layer = [{'weights': [random.random() for i in range(self.n_hidden + 1)]}
                        for i in range(self.n_outputs)]
        self.network.append(self.output_layer)

    def get_network(self):
        return self.network
    
    # back propagation
    def __backprop(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * sigmoid_transfer(neuron['output'])
    
    # forward propagation
    def __fwdprop(self, network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    def train(self, train, l_rate, n_epoch):
        for epoch in range(n_epoch):
            sum_error = 0
        for row in train:
            outputs = self.__fwdprop(self.network, row)
            expected = [0 for i in range(self.n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) **
                              2 for i in range(len(expected))])
            self.__backprop(self.network, expected)
            self.__update_weights(self.network, row, l_rate)
        print('>epoch={}, lrate={:.4}, error={:.4}'.format(
            epoch, l_rate, sum_error))
    
    def predict(self, row):
        outputs = self.__fwdprop(self.network, row)
        return outputs.index(max(outputs))

    def __update_weights(self, network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']