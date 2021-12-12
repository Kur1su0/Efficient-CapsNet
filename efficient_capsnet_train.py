#!/usr/bin/env python
# coding: utf-8

# # Efficient-CapsNet Model Train
# 
# In this notebook we provide a simple interface to train Efficient-CapsNet on the three dataset discussed in "Efficient-CapsNet: Capsule Network with Self-Attention Routing":
# 
# - MNIST (MNIST)
# - smallNORB (SMALLNORB)
# - Multi-MNIST (MULTIMNIST)
# 
# The hyperparameters have been only slightly investigated. So, there's a lot of room for improvements. Good luck!
# 
# **NB**: remember to modify the "config.json" file with the appropriate parameters.

# In[75]:



# In[76]:


import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages, plotHistory
from models import EfficientCapsNet
import matplotlib.pyplot as plt
import numpy as np


# In[77]:


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


model_name = 'MNIST'

dataset = Dataset(model_name, config_path='config.json')

model_train = EfficientCapsNet(model_name, mode='train', verbose=True)



dataset_train, dataset_val = dataset.get_tf_data()



history = model_train.train(dataset, initial_epoch=0)



def plot_history(history):
    plt.figure()
    plt.plot(history.history['val_Efficient_CapsNet_accuracy'])
    plt.plot(history.history['Efficient_CapsNet_accuracy'])

    plt.legend(['val_Efficient_CapsNet_accuracy','Efficient_CapsNet_accuracy'])
    plt.grid(True)
    plt.savefig("Efficient_CapsNet_accuracy.png")
    

    plt.figure()
    plt.plot(history.history['val_Efficient_CapsNet_loss'])
    plt.plot(history.history['Efficient_CapsNet_loss'])

    plt.legend(['val_Efficient_CapsNet_loss','Efficient_CapsNet_loss'])
    plt.grid(True)
    plt.savefig("Efficient_CapsNet_loss.png")
    
plot_history(history)
print("acc img and loss img saved")




