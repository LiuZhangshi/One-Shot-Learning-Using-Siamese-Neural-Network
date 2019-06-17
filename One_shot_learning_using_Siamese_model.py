#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras.models import load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model


# In[21]:


input_shape = (105, 105, 1)
base_model = keras.models.Sequential([
            Conv2D(64, (10, 10), activation = 'relu', input_shape = input_shape),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(128, (7, 7), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(128, (4, 4), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Conv2D(256, (4, 4), activation = 'relu'),
            MaxPooling2D(pool_size=(2, 2), strides = 2),

            Flatten(),
            Dense(4096, activation = 'sigmoid')
    ])


# In[3]:


base_model.load_weights('Siamese_NN_weights.h5', by_name=True)
for layer in base_model.layers:
    layer.trainable = False


# In[4]:


base_model.summary()


# In[58]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from os import listdir
import numpy as np


# In[40]:


img1 = "images_background_small1/Greek/character01/0394_01.png"
input_1 = load_img(img1, target_size = (105, 105, 1), grayscale = True)
input_1 = img_to_array(input_1)
input_1 = input_1.astype('float32')
input_1 /= 255
input_1 = expand_dims(input_1, axis = 0)


# In[49]:


def preprocess_chara(chara):
    character = load_img(chara, target_size = (105, 105), grayscale = True)
    character = img_to_array(character)
    character = character.astype('float32')
    character /= 255
    character = expand_dims(character, axis = 0)
    print(character.shape)
    print(character)
    return character


# In[61]:


def load_characters(directory, model):
    character_vec = {}
    for filename in listdir(directory):
        if filename.endswith(".png"):
            path = directory + filename
            character = preprocess_chara(path)
            vec = model.predict(character)
            name = filename
            character_vec[name] = vec 
            print(name)
    return character_vec


# In[76]:


def what_language(filename, character_vec, model):
    character = preprocess_chara(filename)
    vec = model.predict(character)
    min_dist = 100
    for name in character_vec.keys():
        dist = np.linalg.norm(character_vec[name] - vec)
        print(name)
        print(dist)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.17:
        print("I don't know!")
    else:
        print("The character is, " + str(identity))
        print("Distance is " + str(min_dist))
    return min_dist, identity


# In[77]:


folder = 'characters/'
character_vec = load_characters(folder, base_model)


# In[80]:


character_test = '0414_18.png'
what_language(character_test, character_vec, base_model)


# In[ ]:




