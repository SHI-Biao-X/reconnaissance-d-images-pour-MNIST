# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:36:21 2022

@author: Biao
"""

import numpy as np

def shuffle(images_n, labels_oh):
    """
    requir: matrix of normalized images(number of images * size of images)
    requir: matrix of one hot encoding labels(number of items *  (maximum label + 1))
    ensure: matrix of images and list of labels shuffled randomly but in the same ordre
    """
    
    permutation = np.random.permutation(np.shape(images_n)[0])
    
    images_shuffled = images_n[permutation,:]
    labels_shuffled = labels_oh[permutation,:]
    
    return images_shuffled, labels_shuffled

def get_mini_batches(images_n, labels_oh, mini_batch_size):
    """
    requir: matrix of normalized images(number of images * size of images)
    requir: matrix of one hot encoding labels(number of items *  (maximum label + 1))
    requir: size of mini batch
    ensure: list of mini batches of images and labels
    """

    images_shuffled, labels_shuffled = shuffle(images_n, labels_oh)
    
    num_items = np.shape(images_n)[0]

    #number of mini batches
    num_batch = num_items // mini_batch_size

    mini_batches = []

    for i in range(num_batch):
        mini_batch_images = images_shuffled[i*mini_batch_size:(i+1)*mini_batch_size,:]
        mini_batch_labels = labels_shuffled[i*mini_batch_size:(i+1)*mini_batch_size,:]
        mini_batches.append((mini_batch_images, mini_batch_labels))

    #allocate the rest of datas
    if 0 != len(labels_shuffled) % mini_batch_size:
        mini_batch_images = images_shuffled[num_batch*mini_batch_size:,:]
        mini_batch_labels = labels_shuffled[num_batch*mini_batch_size:,:]
        mini_batches.append((mini_batch_images, mini_batch_labels))

    return mini_batches
    
#test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import LoadData 
    import PreProcess
    
    test_images = LoadData.load_test_images()
    test_labels = LoadData.load_test_labels()
    normalize = PreProcess.normalize(test_images)
    one_hot = PreProcess.one_hot(test_labels)
    mini_batches= get_mini_batches(normalize, one_hot, 64)
    
    print(len(mini_batches))
    print(np.shape(mini_batches[-2][0]))
    print(np.shape(mini_batches[-2][1]))
    for i in range(10):
        print (mini_batches[-1][1][-i])
        plt.imshow(mini_batches[-1][0][-i].reshape(28,28), cmap='gray')
        plt.show()
        
    print ('done')