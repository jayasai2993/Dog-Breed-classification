#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install split-folders')


# In[2]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


# In[3]:


directory = r'C:\Users\SAI\Downloads\archive (1)'


# In[4]:


Name=[]
for file in os.listdir(directory):
    Name+=[file]
print(Name)
print(len(Name))


# In[5]:


dataset=[]
testset=[]
count=0

for file in os.listdir(directory):
    path=os.path.join(directory,file)
    t=0
    for im in os.listdir(path):
        image=load_img(os.path.join(path,im), grayscale=False, color_mode='rgb', target_size=(180,180))
        image=img_to_array(image)
        image=image/255.0
        if t<=20:
            dataset+=[[image,count]]
        else:
            testset+=[[image,count]]
        t+=1
    count=count+1


# In[6]:


data,labels0=zip(*dataset)
test,testlabels0=zip(*testset)


# In[7]:


labels1=to_categorical(labels0)
labels=np.array(labels1)


# In[8]:


data=np.array(data)
test=np.array(test)


# In[9]:


trainx,testx,trainy,testy=train_test_split(data,labels,test_size=0.2,random_state=44)


# In[10]:


print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)


# In[11]:


datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")


# In[12]:


pretrained_model3 = tf.keras.applications.DenseNet201(input_shape=(180,180,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model3.trainable = False


# In[13]:


inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(128, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(14, activation='softmax')(x3)
model = tf.keras.Model(inputs=inputs3, outputs=outputs3)


# In[14]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[15]:


his=model.fit(datagen.flow(trainx,trainy,batch_size=32),validation_data=(testx,testy),epochs=30)


# In[23]:


load_img(r"C:\Users\SAI\Pictures\Screenshots\th (3).jpeg",target_size=(180,180))


# In[24]:


image=load_img(r"C:\Users\SAI\Pictures\Screenshots\th (3).jpeg",target_size=(180,180))

image=img_to_array(image) 
image=image/255.0
prediction_image=np.array(image)
prediction_image= np.expand_dims(image, axis=0)


# In[25]:


dog_breeds = [
    'Affenhuahua', 'Afgan Hound', 'Akita', 'Alaskan Malamute', 'American Bulldog',
    'Auggie', 'Beagle', 'Belgian Tervuren', 'Bichon Frise', 'Bocker',
    'Borzoi', 'Boxer', 'Bugg', 'Bulldog'
]

def mapper(value):
    # Check if the value is within the range of the dog breeds list
    if 1 <= value <= len(dog_breeds):
        return dog_breeds[value]
    else:
        return "Unknown"

# Now you can use the mapper function in your code
prediction = model.predict(prediction_image)
value = np.argmax(prediction)
move_name = mapper(value)
print("Prediction is {}.".format(move_name))


# In[26]:


# Now you can use the mapper function in your code
prediction = model.predict(prediction_image)
value = np.argmax(prediction)
move_name = mapper(value)
print("Prediction is {}.".format(move_name))


# In[ ]:




