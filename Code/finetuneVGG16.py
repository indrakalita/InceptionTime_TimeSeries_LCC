#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import keras as k
from keras import *
#import cv2
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model 
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Input
import csv
from keras.models import model_from_json
from keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
print('[INFO] Importing Librairies ...')
import keras
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.activations import relu
from keras.models import Sequential,Model
import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback
#from google.colab.patches import cv2_imshow
import h5py
import cv2
import os
#from tqdm import tqdm
from random import shuffle
import numpy as np
import scipy.io as scio
import scipy.ndimage as im
import imageio
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, RepeatVector
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, ZeroPadding2D, UpSampling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Add, Multiply, Lambda, AveragePooling2D
from keras.layers import concatenate
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, MaxPool2D
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.advanced_activations import PReLU
from keras.activations import linear as linear_activation
from keras import initializers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report,confusion_matrix
from  sklearn.metrics  import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print('[INFO] Done...')


# In[21]:
import tensorflow as tf
print(tf.test.is_gpu_available())

# Data Preprocessing
from PIL import Image

path = "/home/indrajit/Dataset/TemporalDataset/WSD"
fileA = os.listdir(path)
x = []
y = []
vd = []
vl = []
d={'Agriculture':0,'Beach':1,'Forest':2,'Residential':3,'River':4,'Seawater':5,}
for level1 in fileA:
    fileB = os.listdir(os.path.join(path, level1))
    for level2 in fileB:
        fileC = os.listdir(os.path.join(path, level1, level2))
        for level3 in fileC:
            fileD = os.listdir(os.path.join(path, level1,level2,level3))
            x4 = []
            for level4 in fileD:
                x4.append(np.asarray(Image.open(os.path.join(path, level1, level2, level3, level4)).convert('RGB').resize((224, 224))))
                vd.append(np.asarray(Image.open(os.path.join(path, level1, level2, level3, level4)).convert('RGB').resize((224, 224))))
                vl.append(d[level1])
            x.append(x4)
            y.append(d[level1])
x = np.asarray(x)/255.0
y = np.asarray(y)
vd = np.asarray(vd)/255.0
vl = np.asarray(vl)



# In[22]:


# splitting the dataset into train and test for training the transfer model by freezing the convolutional layers of the model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_val, Y_train, Y_val = train_test_split(vd, vl, test_size=0.1, random_state=42)
Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))
Y_train = to_categorical(Y_train, 6)
Y_val = to_categorical(Y_val, 6)
X_train.shape,X_val.shape,Y_train.shape,Y_val.shape




# Finetuning for VHR dataset
base_model = VGG16(weights='imagenet', include_top=True)
#base_model.summary()
# Freeze all convolution blocks exect the flatten layers in the network
for layer in base_model.layers[:11]:
    layer.trainable = False
for layer in base_model.layers[12:]:
    layer.trainable = True

# create a fully connected network to attatch to the last but one layer of the VGG16
x = base_model.layers[-2].output
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
x = Dense(64, activation='relu')(x)
x = Dense(6, activation='softmax')(x) # Softmax for multiclass
transfer_model = Model(inputs=base_model.input, outputs=x)

# Make sure you have frozen the correct layers
#for i, layer in enumerate(transfer_model.layers):
 #   print(i, layer.name, layer.trainable)


# compile the model 
learning_rate= 0.0001
transfer_model.compile(loss="binary_crossentropy",
                       optimizer=optimizers.Adam(lr=learning_rate),
                       metrics=["accuracy"])


# In[28]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#with tf.device('/cpu:0'):
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('transfer_model_WSD.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = transfer_model.fit(X_train, Y_train,
                             batch_size = 64, epochs=15,
                             validation_data=(X_val,Y_val)) #train the model


#transfer_model.save("transfer_model_WSD.h5")





