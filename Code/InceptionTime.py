#!/usr/bin/env python
# coding: utf-8

# In[25]:


print('[INFO] Importing Librairies ...')
import keras
from keras import layers
from keras.layers import Conv2D, Conv1D, BatchNormalization, Activation, MaxPooling2D, ConvLSTM2D, MaxPooling1D, LSTM
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
from keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, Dense, Flatten 
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
from sklearn.metrics  import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from PIL import Image
from keras.layers import SeparableConv2D, MaxPooling2D
print('[INFO] Done...')


# In[26]:


import tensorflow as tf
print(tf.test.is_gpu_available())


# In[27]:


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



# Load the best model
from keras.models import load_model
 
# load model
model1 = load_model('/home/indrajit/Indicon/transfer_model_WSD.h5')
# FEature extraction from VHR without finetuning
base_model = Model(inputs=model1.inputs, outputs=model1.layers[-2].output)
base_model.summary()
#encoder_data=model.predict(x1)


# In[32]:


from sklearn.model_selection import train_test_split
x1 = x[:,0,:,:,:]
x2 = x[:,1,:,:,:]
x3 = x[:,2,:,:,:]
x4 = x[:,3,:,:,:]
x5 = x[:,4,:,:,:]
print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

X1 = base_model.predict(x1)
X2 = base_model.predict(x2)
X3 = base_model.predict(x3)
X4 = base_model.predict(x4)
X5 = base_model.predict(x5)
print(X1.shape,X2.shape,X3.shape,X4.shape,X5.shape)


# In[33]:


X = []
X.append(X1)
X.append(X1)
X.append(X1)
X.append(X1)
X.append(X1)
X = np.asarray(X)
train_data = np.einsum('ijk->jik', X)

## For Inception module 7X7 features
#train_data = np.concatenate((X1,X2,X3,X4,X5),axis = 3)
print(train_data.shape)
#train_label = np.concatenate((y,y,y,y,y), axis = 3)
#print(train_label.shape)


# In[35]:


# Inception time module with 64 features


# In[36]:


def inception_module(layer_in):
  # 10x conv
  conv1 = Conv1D(32,kernel_size=1, padding='same', activation='relu')(layer_in)
  conv1 = Conv1D(32,kernel_size=10, padding='same', activation='relu')(conv1)
  
  # 30x conv
  conv3 = Conv1D(32,kernel_size=1, padding='same', activation='relu')(layer_in)
  conv3 = Conv1D(32,kernel_size=20, padding='same', activation='relu')(conv3) 
  
  # 50x conv
  conv5 = Conv1D(32,kernel_size=1, padding='same', activation='relu')(layer_in)
  conv5 = Conv1D(32,kernel_size=30, padding='same', activation='relu')(conv5)

  # 3x3 max pooling
  pool = Conv1D(32,kernel_size=1, padding='same', activation='relu')(layer_in)
  pool = MaxPooling1D(3, strides=1, padding='same')(pool)

  # concatenate filters, assumes filters/channels last
  layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
  return layer_out


# In[37]:


from keras.layers import add

def residual_block(layer_in):

  merged_layer =  Conv1D(128,kernel_size=1, padding='same', activation='relu')(layer_in)

  # add inception module
  layer = inception_module(layer_in)
  #layer = inception_module(layer)
  #layer = inception_module(layer)


  return add([layer, merged_layer])


# In[38]:


def inception_model(n_timesteps,n_features,n_outputs):
    visible = Input(shape=(n_timesteps,n_features))

  # # add inception module
  # layer = inception_module(visible)
  # layer = inception_module(layer)
  # layer = inception_module(layer)

    layer = residual_block(visible)
    #layer = residual_block(layer)
    layer = GlobalAveragePooling1D(data_format='channels_last')(layer)
    #layer = Dense(64)(layer)
    layer = Dense(n_outputs, activation='softmax')(layer)
  # create model
    model = Model(inputs=visible, outputs=layer)
  # summarize model
    model.summary()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[52]:


def lstmModel(n_timesteps, n_features,n_outputs):
  # the input shape argument that expects a tuple containing the number of time steps and the number of features.

    model = Sequential()
    model.add(LSTM(32,input_shape=(n_timesteps,n_features)))
    model.add(Dense(24))
    model.add(Dense(n_outputs, activation='softmax'))
    print(model.summary())
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


# In[56]:


n_timesteps, n_features, n_outputs = train_data.shape[1], train_data.shape[2], 6
print(f'Timesteps : {n_timesteps} , Feature Length : {n_features} , Output Length : {n_outputs}')
model = inception_model(n_timesteps, n_features,n_outputs)
#model = lstmModel(n_timesteps, n_features,n_outputs)


# In[40]:


# Inception time with 64 features end


# In[65]:


# splitting the dataset into train and test for training the transfer model by freezing the convolutional layers of the model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, Y_train, Y_test = train_test_split(train_data, y, test_size=0.5, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
Y_val = np.asarray(Y_val).astype('float32').reshape((-1,1))
Y_train = to_categorical(Y_train, 6)
Y_test = to_categorical(Y_test, 6)
Y_val = to_categorical(Y_val, 6)
print(X_train.shape,X_val.shape,X_test.shape,Y_train.shape,Y_val.shape,Y_test.shape)


# In[66]:


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#with tf.device('/cpu:0'):
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('InceptionTime_64feature_WSD.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, Y_train, epochs =15, validation_data=(X_val, Y_val), batch_size=1024, callbacks=[es, mc])


# In[67]:


#Testing of the classification model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
xx = np.argmax(Y_test, axis=1)
pred = model.predict(X_test)
yy = np.argmax(pred, axis=1)
print(classification_report(xx,yy))


# In[ ]:




