import json
import csv
import numpy as np

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from matplotlib import pyplot as plt
from PIL import Image

from sklearn.preprocessing import OneHotEncoder

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

def ConvBlock(layers, model, filters):
	for i in range(layers): 
		model.add(ZeroPadding2D((1,1)))
		model.add(Convolution2D(filters, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

def FCBlock(model):
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))

vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))

def vgg_preprocess(x):
	x = x - vgg_mean     # subtract mean
	return x[:, ::-1]    # reverse axis bgr->rgb

def VGG_16():
	model = Sequential()
	model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))
	ConvBlock(2, model, 64)
	ConvBlock(2, model, 128)
	ConvBlock(3, model, 256)
	ConvBlock(3, model, 512)
	ConvBlock(3, model, 512)
	model.add(Flatten())
	FCBlock(model)
	FCBlock(model)
	model.add(Dense(1000, activation='softmax'))
	return model

model = VGG_16()

fpath = get_file('vgg16.h5', 'vgg16.h5', cache_subdir='models') # See: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

model.load_weights(fpath)

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical', target_size=(224,224)):
	return gen.flow_from_directory(dirname, target_size=target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

val_batches = get_batches('n11669921/sample/valid', shuffle=False, batch_size=64)
batches = get_batches('n11669921/sample/train', shuffle=False, batch_size=64)

def onehot(x): 
	return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)

# Fine-tuning

model.pop()
for layer in model.layers: 
	layer.trainable = False

model.add(Dense(121, activation='softmax'))

def fit_model(model, batches, val_batches, nb_epoch=1):
	model.fit_generator(batches, samples_per_epoch=batches.N, nb_epoch=nb_epoch, validation_data=val_batches, nb_val_samples=val_batches.N)

opt = RMSprop(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

fit_model(model, batches, val_batches, nb_epoch=2)

preds = model.predict_classes(val_data, batch_size=64)
probs = model.predict_proba(val_data, batch_size=64)[:,0]

layers = model.layers
# Get the index of the first dense layer
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# Set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]: 
	layer.trainable = True

K.set_value(opt.lr, 0.0001)
fit_model(model, batches, val_batches, 3)

model.save_weights('models/finetune2.h5')

