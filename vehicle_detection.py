import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape


from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box


keras.backend.set_image_dim_ordering('th')

model = Sequential()
model.add(Convolution2D(16, (3, 3),input_shape=(3,448,448),padding='same',strides=(1,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Convolution2D(64,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Convolution2D(128,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Convolution2D(256,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Convolution2D(512,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
model.add(Convolution2D(1024,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,(3,3) ,padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))


model.summary()