import tensorflow as tf
from tensorflow import keras 
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import RandomCrop, RandomFlip, RandomRotation
from keras.models import Sequential
from keras import initializers 
from keras import regularizers 
from keras import constraints
from keras import losses 
from keras import optimizers 
from keras import metrics 
from keras.applications import * #Efficient Net included here
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

def build_model(input_dimensions, efficientnet_density):    
    # weights initializers
    zeros_init = initializers.Zeros() 
    ones_init = initializers.Ones() 
    constant_init = initializers.Constant(value = 1) 
    random_init = initializers.RandomNormal(mean=0.0, stddev = 0.05, seed = None) 
    
    # weights constraints
    # weight less than or equal to max_value
    max_constrain = constraints.MaxNorm(max_value = 10, axis = 0) 
    # weight between min_value and max_value
    min_max_constrain = constraints.MinMaxNorm(min_value = 0.0, max_value = 10.0, rate = 1.0, axis = 0)
    
    # regularizers
    # tools used to modify net weights basing on the errors
    # reduce_sum - returns sum of vector elements along given axis 
    # The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))
    # The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))

    l1_regularizer = regularizers.L1(0.01)
    l2_regularizer = regularizers.L2(0.01)

    # optimizers
    learning_rate = 0.01
    # SGD - high variance - fluctuations of obj. func. values
    # good to decrease learning rate during learning process to avoid oscillations

    # ADAM, AdaGrad - modify lrs diffrently for each neuron
    # ADA dla danych ktorych czesc cech ma niewielka reprezentacje w danej
    # a jednoczesnie maja cechy ktorych jest duzo w danej (Dense, Sparse)
    opt_sgd = optimizers.SGD(lr=learning_rate)
    # opt = "adagrad"
    # opt = "sgd"
    opt = "adam"

    # loss functions
    loss_fn = "binary_crossentropy"
    # loss_fn = "mean_squared_error"

    # metrics
    metrics = 'accuracy'

    # layers definition    
    # pooling layers
    max_pool_2d = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
    max_pool_2d_gap = GlobalMaxPooling2D(name="gap")
    
    # output layer
    output_layer = Dense(1, activation = 'sigmoid')

    # convolutional base
    # Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, ... up to  7
    # Higher the number, the more complex the model is. and the larger resolutions it  can handle, 
    # but  the more GPU memory it will need# loading pretrained conv base model
    # #input_shape is (height, width, number of channels) for images

    # Density of EfficientNet
    if efficientnet_density == 1:
        conv_base = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 2:
        conv_base = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 3:
        conv_base = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 4:
        conv_base = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 5:
        conv_base = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 6:
        conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))
    elif efficientnet_density == 7:
        conv_base = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(input_dimensions[0], input_dimensions[1], 3))


    # model definition
    model = Sequential()
    model.add(conv_base)
    model.add(max_pool_2d_gap)

    #avoid overfitting
    # model.add(layers.Dropout(dropout_rate=0.2, name="dropout_out"))
    model.add(output_layer)
    conv_base.trainable = False

    # model compilation
    model.compile(
        optimizer=opt,
        loss = loss_fn,
        metrics = metrics
        )
    
    return (model, opt, loss_fn, metrics)
