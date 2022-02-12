# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:35:19 2022

@author: Biao
"""

import numpy as np

#layers size(with 2 hidden layers in this exemple)
layers_size = [784,20,20,10] 

def forward(images_n, parameters):
    """
    requir: matrix of normalized images (number of images * size of images)
    requir: parameters ((number of layers-1) * (matrix of weights, vertor of bias))
    ensure: list of interal variables (Si = Xi * Wi + bi, Zi = sigma(Si))
    ensure: matrix of predictions (number of images * 10)
    """
    
    interal_variables = []
    
    wi,bi = parameters[0]
    
    Si = np.dot(images_n,wi) + bi #S = X * W + b
    Zi = activation_fonction(Si)
    
    interal_variables.append((Si, Zi))
    for i in range(1, len(parameters) - 1):
        wi,bi = parameters[i]
        Si = np.dot(Zi, wi) + bi
        Zi = activation_fonction(Si)
        interal_variables.append((Si,Zi))
    
    wi,bi = parameters[-1]
    Si = np.dot(Zi, wi) + bi
    predictions = softmax(Si)
    return interal_variables, predictions
        
def activation_fonction(S):
    """
    requir: matrix S
    ensure: matrix Z so that Z = sigmoid(S)
    """
    return 1 / (1 + np.exp(-S))
        
def softmax(Z):
    """
    requir: matrix Z
    ensure: matrix y so that y = softmax(O)
    """
    y = np.empty((np.shape(Z)[0],np.shape(Z)[1]))
    for i in range(np.shape(Z)[0]):
        y[i] = np.exp(Z[i]) / np.sum(np.exp(Z[i]))
    return y

#test
if __name__ == '__main__':
    import LoadData 
    import PreProcess
    import Initialization
    
    test_images = LoadData.load_test_images()[:100,:]
    test_labels = LoadData.load_test_labels()[:100]
    normalize = PreProcess.normalize(test_images)
    one_hot = PreProcess.one_hot(test_labels)
    
    parameters = Initialization.init_parameters(layers_size)
    
    interal_variables, predictions = forward(normalize, parameters)
    
    print(predictions[0])
        
    print ('done')
    