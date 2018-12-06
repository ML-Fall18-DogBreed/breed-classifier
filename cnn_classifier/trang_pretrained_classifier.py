#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import cv2
from create_model_trang import create_cnn_model_trang
from keras.callbacks import ModelCheckpoint  

from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from os import makedirs
from os.path import expanduser, exists, join


# In[2]:


get_ipython().system('ls ../keras-pretrained-models/')

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
    
get_ipython().system('cp ../keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')
get_ipython().system('cp ../keras-pretrained-models/resnet50* ~/.keras/models/')


# In[3]:


train_folder = '../train/'
test_folder = '../test/'


# In[4]:


train_dogs = pd.read_csv('../labels.csv')
train_dogs.head()


# In[5]:


train_dogs['image_path'] = train_dogs.apply( lambda x: (train_folder + x["id"] + ".jpg" ), axis=1)


# In[6]:


train_dogs.head(10)


# In[7]:


# Get an array from the image paths to train on
train_data = np.array([img_to_array(load_img(img, target_size=(224, 224))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')


# In[8]:


# Split the data into train and validation. Since we only have train and validation folders, need to divide train into training and validation sets. 
# Save validation folder for later testing
x_train, x_validation, y_train, y_validation = train_test_split(train_data, train_dogs["breed"], test_size=0.2, stratify=np.array(train_dogs["breed"]), random_state=1234)


# In[9]:


print ('x_train shape = ', x_train.shape)
print ('x_validation shape = ', x_validation.shape)


# In[14]:


one_hot = pd.get_dummies(train_dogs["breed"], sparse = True)
one_hot_labels = np.asarray(one_hot)


# In[10]:


# Need to convert the train and validation labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).as_matrix()


# In[11]:


# Create train generator.
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   # zoom_range = 0.3, 
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, 
                                   horizontal_flip = 'true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=False, batch_size=20, seed=10)


# In[12]:


# Create validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=False, batch_size=20, seed=10)


# In[15]:


model = create_cnn_model_trang()


# In[16]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[17]:


checkpointer = ModelCheckpoint(filepath='out_trang/weights.newbestaugmented.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
batch_size = 20
model.fit_generator(
        train_generator,
        steps_per_epoch=2000// batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=800 // batch_size,
        callbacks=[checkpointer])


# In[13]:


# Get the InceptionV3 model so we can do transfer learning
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(224, 224, 3))
# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer and a logistic layer with 20 classes 
#(there will be 120 classes for the final submission)
x = Dense(512, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)
# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
# first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    
checkpointer = ModelCheckpoint(filepath='out_trang/weights.pretrained.hdf5', 
                               verbose=1, save_best_only=True)
# Compile with Adam
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit_generator(train_generator,
                      steps_per_epoch = 409,
                      validation_data = val_generator,
                      validation_steps = 102,
                      epochs = 20,
                      verbose = 2, callbacks=[checkpointer])


# In[17]:


# Use the sample submission file to set up the test data - x_test
test_data = pd.read_csv('../sample_submission.csv')
# Create the x_test
x_test = []
for i in tqdm(test_data['id'].values):
    img = cv2.imread('../test/{}.jpg'.format(i))
    x_test.append(cv2.resize(img, (299, 299)))
# Make it an array
x_test = np.array(x_test, np.float32) / 255.


# In[ ]:


# Predict x_test
predictions = model.predict(x_test, verbose=2)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values


# In[ ]:


# Create the submission data.
submission_results = pd.DataFrame(predictions, columns = col_names)
# Add the id as the first column
submission_results.insert(0, 'id', test_data['id'])
# Save the submission
submission_results.to_csv('submission.csv', index=False)

