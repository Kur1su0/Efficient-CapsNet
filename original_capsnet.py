#!/usr/bin/env python
# coding: utf-8

# # Original CapsNet Model Train
# 
# In this notebook we provide a simple interface to train the original CapsNet model described in "Dynamic routinig between capsules". The model is copycat of the original Sara's repository (https://github.com/Sarasra/models/tree/master/research/capsules). <br>
# However, if you really reach 99.75, you've got to buy me a drink :)

# In[10]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[11]:


import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages
from models import CapsNet
import matplotlib.pyplot as plt


# In[12]:


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# In[13]:


# some parameters
model_name = 'MNIST' # only MNISTisMHD is available
n_routing = 3


# # 1.0 Import the Dataset

# In[14]:


dataset = Dataset(model_name, config_path='config.json') # MHD


# ## 1.1 Visualize imported dataset

# In[15]:


n_images = 20 # number of images to be plotted
plotImages(dataset.X_test[:n_images,...,0], dataset.y_test[:n_images], n_images, dataset.class_names)


# # 2.0 Load the Model

# In[16]:


dataset_train, dataset_val = dataset.get_tf_data()
model_train = CapsNet(model_name, mode='train', verbose=True, n_routing=n_routing)


# # 3.0 Train the Model

# In[17]:


history = model_train.train(dataset, initial_epoch=0)


# In[18]:


def plot_history(history):
    plt.plot(history.history['val_Original_CapsNet_accuracy'])
    plt.plot(history.history['Original_CapsNet_accuracy'])

    plt.legend(['val_Original_CapsNet_accuracy','Original_CapsNet_accuracy'])
    plt.grid(True)
    plt.show()


    plt.plot(history.history['val_Original_CapsNet_loss'])
    plt.plot(history.history['Original_CapsNet_loss'])

    plt.legend(['val_Original_CapsNet_loss','loss'])
    plt.grid(True)
    plt.show()
plot_history(history)


# In[ ]:





# In[ ]:




