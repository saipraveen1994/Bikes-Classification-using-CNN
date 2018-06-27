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
TEST_DIR = argList[1]
IMG_SIZE = 64 # Compressing Image Size to 64*64
LR = 5e-4  # Learning Rate
MODEL_NAME = 'classifybikes-{}-{}.model'.format(LR, 'mymodel')

############################################################################
# A function to process the testdata
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]  # Extracting the name of the image
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), np.array(img_num)]) # Appending testimages and names of the images
        shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
        
############################################################################
# Building the model
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')  # Convolution Layer-1 with 5 Filters of size 32*32
convnet = max_pool_2d(convnet, 5)  # Max Pooling with block stride of 5
convnet = conv_2d(convnet, 64, 5, activation='relu')  # Convolution Layer-2 with 5 Filters of size 64*64
convnet = max_pool_2d(convnet, 5)  # Max pooling with block stride of 5
convnet = conv_2d(convnet, 32, 5, activation='relu')  # Convolution Layer-3 with 5 Filters of size 32*32
convnet = max_pool_2d(convnet, 5)  # Max pooling with block stride of 5
convnet = fully_connected(convnet, 1024, activation='relu') # Fully Connected Layer-4 with 1024 neurons
convnet = dropout(convnet, 0.4) # Dropout rate set to 0.4 
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')

############################################################################
# Modelling the test data
test_data = create_test_data()
test = test_data
test_x = np.array([i[0] for i in test])
test_num = [i[1] for i in test]

############################################################################
# Evaluating the model
model.load(MODEL_NAME)
prediction = model.predict(test_x)
count = 0
for i in prediction:
    if float(i[0]) > float(i[1]) :
        print("For the bike: " + str(test_num[count]) + " " + "the model says that the image has characteristics of mountainbike with a confidence level of : " + " " + str(i[0]))
    else:
        print("For the bike: " + str(test_num[count]) + " " + "the model says that the image has characteristics of roadbike with a confidence level of : " + " " + str(i[1]))
    count += 1

############################################################################

        
    
