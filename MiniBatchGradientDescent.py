# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 00:33:21 2022

@author: Biao
"""

import Initialization
import MiniBatch
import Forward
import Backward
import numpy as np
import matplotlib.pyplot as plt
import math

def mini_batch_gradient_descent(train_images, train_labels, gamma0, num_epoches, mini_batch_size, layers_size):
    """
    requir: matrix of train images(number of images * size of images)
    requir: vector of train labels(number of items)
    requir: learning rate initial: gamma0
    requir: number of epoches
    requir: mini batch size
    requir: layer size
    ensure: list of parameters final
    ensure: list of loss
    """
    
    #normalize matrix of train images and one-hot coding train labels
    images_n = PreProcess.normalize(train_images)
    labels_oh = PreProcess.one_hot(train_labels)
    
    parameters = Initialization.init_parameters(layers_size)
    L_loss = [] #list of loss for each epoch
    
    for epoch in range(1, num_epoches+1):
        gamma = gamma0/(epoch)**0.6
        mini_batches = MiniBatch.get_mini_batches(images_n, labels_oh, mini_batch_size)
        for mini_batch_images_n, mini_batch_labels_oh in mini_batches:
            interal_variables, predictions = Forward.forward(mini_batch_images_n, parameters)
            parameters_grad = Backward.backward(mini_batch_images_n, mini_batch_labels_oh, \
                                                predictions, interal_variables, parameters)
            parameters = update_parameters(gamma, parameters, parameters_grad)
        loss = np.mean(Backward.loss(mini_batch_labels_oh, predictions))
        L_loss.append(loss)
        if(epoch % 1000 == 0):
            print("number of epoches: %d loss: %f"% (epoch, loss))

    return parameters, L_loss
            
            
def update_parameters(gamma, parameters, parameters_grad):
    """
    requir: learning rate: gamma
    requir: list of parameters
    requir: list of parameter gradients
    ensure: update parameters
    """
    for i in range(len(parameters)):
        Wi, bi = parameters[i]
        Wi_grad, bi_grad = parameters_grad[i]
        parameters[i] = (Wi - gamma * Wi_grad, bi - gamma * bi_grad)
    
    return parameters
        
def plot_loss(L_loss):
    """
    requir: list of loss
    ensure: plot loss fonction
    """
    
    plt.plot(L_loss)
    plt.xlabel("number of epoches") 
    plt.ylabel("loss")
    plt.show()
    
    return 

def score(parameters, test_images, test_labels):
    """
    requir: list of parameters
    requir: matrix of test images
    requir: vector of test labels
    ensure: accuracy score
    """
    
    #normalize matrix of test images and one-hot coding test labels
    images_n = PreProcess.normalize(test_images)
    labels_oh = PreProcess.one_hot(test_labels)
    
    interal_variables, predictions = Forward.forward(images_n, parameters)
    
    return np.mean(np.equal(np.argmax(predictions, axis = 1), np.argmax(labels_oh, axis = 1)))

def illustrate_predictions(num_exemples, parameters, test_images):
    """"
    requir: number of exemples
    requir: list of parameters
    requir: matrix of test images
    ensure: illustrate some exemples of predictions
    """
    images_n = PreProcess.normalize(test_images)
    
    id_exemples = np.random.choice(np.shape(test_images)[0], size = num_exemples, replace = False)
    
    size_subplot = math.ceil(math.sqrt(num_exemples))
    
    for i in range(num_exemples):
        plt.subplot(size_subplot,size_subplot,i + 1)
        interal_variables, predictions = Forward.forward(images_n[id_exemples[i],:], parameters)
        plt.title("predition: "+ str(np.argmax(predictions)))
        plt.imshow(test_images[id_exemples[i],:].reshape(28,28), cmap='gray')
    
    plt.show()
    return

#test
if __name__ == '__main__':
    import LoadData
    import PreProcess
    
    gamma0 = 0.05
    num_epoches = 1000
    mini_batch_size = 64
    #layers size(with 2 hidden layers in this exemple)
    layers_size = [784,30,20,10]
    
    train_images = LoadData.load_train_images()[:60000,:]
    train_labels = LoadData.load_train_labels()[:60000]
    test_images = LoadData.load_test_images()[:10000,:]
    test_labels = LoadData.load_test_labels()[:10000]
    
    parameters, L_loss = mini_batch_gradient_descent(train_images, train_labels, gamma0, num_epoches, mini_batch_size, layers_size)
    
    #print(L_loss)
    
    plot_loss(L_loss)
    
    print("accuracy score is",score(parameters, test_images, test_labels))
    
    illustrate_predictions(16, parameters, test_images)