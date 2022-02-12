# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 23:54:49 2022

@author: Biao
"""


# train-images-idx3-ubyte: training set images
# train-labels-idx1-ubyte: training set labels
# t10k-images-idx3-ubyte:  test set images
# t10k-labels-idx1-ubyte:  test set labels

# The training set contains 60000 examples, and the test set 10000 examples.

# The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label

# The labels values are 0 to 9.
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel

# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label

# The labels values are 0 to 9.
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  10000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel

# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
#

import numpy as np
import struct

def read_idx1_ubyte(idx1_ubyte_file):
    """
    require:address of idx1 ubyte file
    ensure: vector of labels
    """
    
    bin_data = open(idx1_ubyte_file, 'rb').read()

    #read head information
    offset = 0
    fmt_head = ">ii" #format of head(2 integers)
    magic_number, num_items = struct.unpack_from(fmt_head, bin_data, offset)
    print ("magic number: %d, number of items: %d" % (magic_number, num_items) )

    #read label information
    offset += struct.calcsize(fmt_head)
    #print(offset)
    fmt_label = '>B'   #format of labels '>B'
    labels = np.empty(num_items, dtype = np.uint8)
    for i in range(num_items):
        if (i + 1) % 10000 == 0:
            print ("have read %d labels" % (i + 1))
        labels[i] = struct.unpack_from(fmt_label, bin_data, offset)[0]
        #print(labels[i])
        offset += struct.calcsize(fmt_label)
    
    print("read successfully")
    #print(labels)
    
    return labels

#read_idx1_ubyte("./data/t10k-labels.idx1-ubyte")

def read_idx3_ubyte(idx3_ubyte_file):
    """
    require:address of idx3 ubyte file
    ensure: matrix of images(number of images * size of image)
    """
    
    bin_data = open(idx3_ubyte_file, 'rb').read()

    #read head information
    offset = 0
    fmt_head = ">iiii" #format of head(4 integers)
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_head, bin_data, offset)
    print ("magic number: %d, number of images: %d, image size: %d*%d" % (magic_number, num_images, num_rows, num_cols) )

    #read image information
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_head)
    fmt_image = '>' + str(image_size) + 'B'   #format of images '>784B'
    images = np.empty((num_images, image_size), dtype = np.uint8)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ("have read %d images" % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    
    print("read successfully")
    #print(images)
    #print(np.shape(images))
    
    return images

#read_idx3_ubyte("./data/t10k-images.idx3-ubyte")

def load_train_images(idx3_ubyte_file = "./data/train-images.idx3-ubyte"):
    return read_idx3_ubyte(idx3_ubyte_file)

def load_test_images(idx3_ubyte_file = "./data/t10k-images.idx3-ubyte"):
    return read_idx3_ubyte(idx3_ubyte_file)

def load_train_labels(idx1_ubyte_file = "./data/train-labels.idx1-ubyte"):
    return read_idx1_ubyte(idx1_ubyte_file)

def load_test_labels(idx1_ubyte_file = "./data/t10k-labels.idx1-ubyte"):
    return read_idx1_ubyte(idx1_ubyte_file)

#test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    train_images = load_train_images() 
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    
    for i in range(10):
        print (train_labels[i])
        plt.imshow(train_images[i].reshape(28,28), cmap='gray')
        plt.show()
    print ('done')