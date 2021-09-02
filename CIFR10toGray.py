#load the CIFAR10 data

from tensorflow.keras.datasets import cifar10

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

(x_train,_),(x_test,_) = cifar10.load_data()

#use format "chennels_last"

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]
print("img_rows:{},img_cols:{},channels:{}".format(img_rows,img_cols,channels))
imgs_dir = 'saved_images'
save_dir = os.path.join(os.getcwd(),imgs_dir)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# display the 1st 100 input images (color and gray)

imgs = x_test[:100]
print("imgs.shape:{}".format(imgs.shape))
imgs = imgs.reshape((10,10,img_rows,img_cols,channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color images (Ground  Truth)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % imgs_dir)
plt.show()

imgs_gray = rgb2gray(x_train)
imgs = imgs_gray[:100]
print("imgs_gray:{}".format(imgs.shape))
imgs = imgs.reshape((10, 10, img_rows, img_cols))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test gray images (Input)')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % imgs_dir)
plt.show()