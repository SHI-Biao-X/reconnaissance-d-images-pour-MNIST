# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:48:00 2022

@author: Biao
"""

import numpy as np

def one_hot(labels):
    """
    requir: vector of integer 
    ensure: matrix of one hot encoding of this list(number of items * 10)
    """
    return np.eye(10)[labels]

def normalize(images):
    """
    requir: matrix of image pixel(0-255)
    ensure: matrix normalize of image pixel(0-1)
    """
    return (images - np.mean(images)) / np.std(images)
    
#test
if __name__ == '__main__':
    import LoadData 
    
    test_labels = LoadData.load_test_labels()
    one_hot = one_hot(test_labels)
    test_images = LoadData.load_test_images()
    normalize = normalize(test_images)
    for i in range(10):
        print (test_labels[i])
        print (one_hot[i])
        #print (test_images[i])
        #print (normalize[i])
    print ('done')
    