# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:54:11 2022

@author: Biao
"""

import numpy as np

#layers size(with 2 hidden layers in this exemple)
layers_size = [784,20,20,10] 

def init_parameters(layers_size):
    """
    requir: list of layers size
    ensure: list of initialized parameters
    """

    parameters = []

    for i in range(len(layers_size) - 1):
        parameters.append(( init_weights(layers_size[i], layers_size[i+1]),\
                           init_bias(layers_size[i+1]) ))

    return parameters

def init_weights(size_in, size_out):
    """
    requir: input layers size 
    requir: output layers size
    ensure: initialized matrix of weight with Xavier initialization
    """

    return np.random.randn(size_in, size_out)  / size_in
    
def init_bias(size_out):
    """ 
    requir: output layers size
    ensure: initialized matrix of bias
    """

    return np.zeros(size_out)

#test
if __name__ == '__main__':
    parameters = init_parameters(layers_size)
    
    print(len(parameters))
    for i in range(len(parameters)):
        for j in [0,1]:
            print(i,j,np.shape(parameters[i][j]))