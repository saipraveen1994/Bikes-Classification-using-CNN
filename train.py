import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import tflearn             # dealing with making CNN model
import sys                 # Package to deal with commandline arguments
import tensorflow as tf    # dealing with making CNN model
from os import listdir     # dealing with importing images
from os.path import isfile, join
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

############################################################################
argList = sys.argv
TRAIN_DIR = argList[1]
IMG_SIZE = 64  # Compressing Image Size to 64*64
LR = 5e-4  # Learning Rate
MODEL_NAME = 'classifybikes-{}-{}.model'.format(LR, 'mymodel')


############################################################################
# A function to extract the softnet(Outputs)
def label_img(img):
    word_label = img.split('_')[-3]
    # conversion to one-hot array [mountainbike , roadbike]
    if word_label == 'mountain': return [1,0]
    elif word_label == 'road': return [0,1]

############################################################################
# A function to process the train data
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # Resizing the image to 64*64
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

#############################################################################
# Building the model
tf.reset_default_graph()
train_data = create_train_data()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu') # Convolution Layer-1 with 5 Filters of size 32*32
convnet = max_pool_2d(convnet, 5)  # Max Pooling with block stride of 5
convnet = conv_2d(convnet, 64, 5, activation='relu') # Convolution Layer-2 with 5 Filters of size 64*64
convnet = max_pool_2d(convnet, 5)  # Max pooling with block stride of 5
convnet = conv_2d(convnet, 32, 5, activation='relu')  # Convolution Layer-3 with 5 Filters of size 32*32
convnet = max_pool_2d(convnet, 5) # Max pooling with block stride of 5
convnet = fully_connected(convnet, 1024, activation='relu') # Fully Connected Layer-4 with 1024 neurons
convnet = dropout(convnet, 0.4) # Dropout rate set to 0.4 
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')

##############################################################################
# Modelling the training data
train = train_data
X = np.array([i[0] for i in train])
Y = [i[1] for i in train]
model.fit({'input': X}, {'targets': Y}, n_epoch=17, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
##############################################################################
