# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 20:55:50 2022

@author: Biao
"""

import numpy as np
import Forward

def backward(images_n, labels_oh, predictions, interal_variables, parameters):
    """
    requir: matrix of normalized images(number of images * size of images)
    requir: matrix of one hot encoding labels(number of items * 10)
    requir: matrix of predictions (number of images * 10)
    requir: list of interal variables (Si = Xi * Wi + bi, Zi = sigma(Si))
    requir: list of parameters
    ensure: list of parameter gradients
    """
    
    #calcul the gradient of the last Wi et bi
    temp = predictions - labels_oh
    Si,Zi = interal_variables[-1]
    Wi_grad = np.dot(temp.T, Zi).T
    bi_grad = np.dot(temp.T, np.ones(np.shape(Zi)[0]).T).T
    
    parameters_grad = [(Wi_grad, bi_grad)]
    
    #calcul the gradient of middle Wi et bi
    for i in range(len(interal_variables)-2, -1, -1):
        Wi2 = parameters[i + 2][0] #Wi2 = W_(i+2)
        #Si = S_(i+1)
        temp = np.dot(temp, Wi2.T) * deri_activation_fonction(Si)
        #Si = S_i; Zi = Z_i
        Si,Zi = interal_variables[i]
        Wi_grad = np.dot(temp.T, Zi).T
        bi_grad = np.dot(temp.T, np.ones(np.shape(Zi)[0]).T).T
        
        parameters_grad.insert(0, (Wi_grad, bi_grad))
        
    #calcul the gradient of the first Wi et bi
    Wi2 = parameters[1][0] #W1
    temp = np.dot(temp, Wi2.T) * deri_activation_fonction(Si)
    Wi_grad = np.dot(temp.T, images_n).T
    bi_grad = np.dot(temp.T, np.ones(np.shape(images_n)[0]).T).T
    
    parameters_grad.insert(0, (Wi_grad, bi_grad))
    
    return parameters_grad
    
def deri_activation_fonction(S):
    """
    requir: matrix S
    ensure: matrix D = sigmoid'(S) = sigmoid(S) * (1 - sigmoid(S))
    """
    
    return Forward.activation_fonction(S) * (1 - Forward.activation_fonction(S))
    
    
def loss(labels_oh, predictions):
    """
    requir: matrix of one hot encoding labels(number of items * 10)
    requir: matrix of predictions (number of images * 10) 
    ensure: vector of cross-entrogy loss 
    """
    
    loss = np.empty(np.shape(labels_oh)[0])
    for i in range(np.shape(loss)[0]):
        loss[i] = -np.log(predictions[i][np.argmax(labels_oh[i])])
        
    return loss

#test
if __name__ == '__main__':
    import LoadData 
    import PreProcess
    import Initialization
    
    #layers size(with 2 hidden layers in this exemple)
    layers_size = [784,20,20,10] 
    
    test_images = LoadData.load_test_images()[:30,:]
    test_labels = LoadData.load_test_labels()[:30]
    normalize = PreProcess.normalize(test_images)
    one_hot = PreProcess.one_hot(test_labels)
    print(np.shape(one_hot))
    
    parameters = Initialization.init_parameters(layers_size)
    
    interal_variables, predictions = Forward.forward(normalize, parameters)
    
    a = np.array([[0,1],\
                  [1,0]])
        
    b = np.array([[0,1],\
                  [0.9,0.9]])
    loss = loss(a, b)
    print(loss)
    
    parameters_grad = backward(normalize, one_hot, predictions, interal_variables, parameters)
    print(len(parameters_grad))
    for i in range(len(parameters_grad)):
        for j in [0,1]:
            print(i,j,np.shape(parameters_grad[i][j]))
        
    print ('done')
    