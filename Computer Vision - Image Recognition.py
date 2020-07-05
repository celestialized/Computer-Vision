#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


mnist = tf.keras.datasets.fashion_mnist


# In[3]:


(training_images, training_labels),(test_images, test_labels) = mnist.load_data()


# In[4]:


training_images = training_images/255
test_images = test_images/255


# In[5]:


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                   ])


# In[6]:


model.compile(optimizer = tf.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


# In[7]:


model.fit(training_images, training_labels, epochs=5)


# In[12]:


print(test_labels[2])


# In[10]:


model.evaluate(test_images, test_labels)


# In[13]:


import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[2])
print(training_labels[2])
print(training_images[2])


# In[ ]:




