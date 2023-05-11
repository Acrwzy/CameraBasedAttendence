##### LIBRARIES #####
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
##### #####

##### PREPROCESSING #####
SIZE = 512           ## tells us the resolution of the image, can also try with 256, 1024, 2048  
train= " "           ## insert directory for the imagees used to train the model
test = " "           ## insert directory for the imagees used to test the model

vgg = VGG16(Input=[SIZE, SIZE, 3], weights='imagenet')
for layer in vgg.layers():
    layer.trainable=False

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
