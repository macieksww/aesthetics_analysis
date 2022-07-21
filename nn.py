import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from import_cv import process_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from keras.models import Sequential 
from keras.layers import Activation, Dense, Dropout, Flatten
from keras import initializers 
from keras import regularizers 
from keras import constraints
from keras import losses 
from keras import optimizers 
from keras import metrics 
from save_train_data import data_saver

def prep_dataset():

    data_dir = "/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432"
    # number of all images in dataset
    num_of_images = process_directory(data_dir)
    print("NUM OF IMAGES")
    print(num_of_images)
    # number of samples used in every training episode
    # batch_size = int(num_of_images/15)
    epochs = 1
    batch_size = 32
    print("BATCH SIZE")
    print(batch_size)
    image_size = (300, 432)
    
    aesthetic = list(os.listdir(data_dir+'/aesthetic'))
    nonaesthetic = list(os.listdir(data_dir+'/nonaesthetic'))
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_size[0], image_size[1]),
        batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_size[0], image_size[1]),
        batch_size=batch_size)
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    class_names = train_ds.class_names
    print("DATA CLASSES:")
    print(class_names)
    
    # prefetch and caching for better dataset performace
    # caching keeps images in caceh after they are loaded
    # from disk during first epoch
    # prefetch overlaps data preprocessing and model execution
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    model = build_model([image_size[0], image_size[1]])[0]
    opt = build_model([image_size[0], image_size[1]])[1]
    loss_fn = build_model([image_size[0], image_size[1]])[2]
    metrics = build_model([image_size[0], image_size[1]])[3]

    model.summary()
    model.fit(train_ds, batch_size = batch_size, validation_data = val_ds, epochs=epochs)

    loss, accuracy = model.evaluate(val_ds, verbose=1) 
    print("LOSS")
    print(loss)
    print("ACCURACY")
    print(accuracy)
    
    model.summary()

    training_saver = data_saver()
    training_saver.save_data("model_params", [opt, loss_fn, metrics, epochs, batch_size])

def build_model(input_dimensions):
    # model = Sequential([
    # keras.layers.Flatten(input_shape=(image_size[0], image_size[1])) # input layer
    # keras.layers.Dense(128, activation='relu') # hidden layer
    # keras.layers.Dense(2, activation='softmax') # output layer (2 nodes - 2 classes)
    # ])
    
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
    learning_rate = 0.5
    # SGD - high variance - fluctuations of obj. func. values
    # good to decrease learning rate during learning process to avoid oscillations

    # ADAM, AdaGrad - modify lrs diffrently for each neuron
    # ADA dla danych ktorych czesc cech ma niewielka reprezentacje w danej
    # a jednoczesnie maja cechy ktorych jest duzo w danej (Dense, Sparse)
    opt_sgd = optimizers.SGD(lr=learning_rate)
    opt = "adam"

    # loss functions
    loss_fn = "binary_crossentropy"

    # metrics
    metrics = 'accuracy'

    # layers definition
    # input_layer = Dense(32, input_shape=(input_dimensions[0], input_dimensions[1], 3, ), 
    # kernel_initializer = ones_init, kernel_regularizer = l1_regularizer, 
    # kernel_constraint = 'MaxNorm', activation = 'relu')
    base = input_dimensions[0] * input_dimensions[1] * 3

    input_layer = Flatten(input_shape=(input_dimensions[0], input_dimensions[1], 3))
    hidden_layer_1 = Dense(512,  activation = 'relu')
    hidden_layer_2 = Dense(256,  activation = 'relu')
    hidden_layer_3 = Dense(64,  activation = 'relu')
    hidden_layer_4 = Dense(16,  activation = 'relu')
    hidden_layer_5 = Dense(8,  activation = 'relu')
    hidden_layer_6 = Dense(2,  activation = 'relu')
    # output_layer = Dense(1, activation = 'sigmoid')
    output_layer = Dense(1, activation = 'softmax')
    
    # model definition
    model = Sequential()
    model.add(input_layer)
    # model.add(Dropout(0.05)) 
    # model.add(hidden_layer) 
    # model.add(Dropout(0.05)) 
    model.add(hidden_layer_1)
    model.add(hidden_layer_2)
    model.add(hidden_layer_3)
    model.add(hidden_layer_4)
    model.add(hidden_layer_5)
    model.add(hidden_layer_6)

    # model.add(Dropout(0.05)) 
    model.add(output_layer)
    
    # model compilation
    
    model.compile(
        optimizer=opt,
        # optimizer = opt_sgd,
        # loss = 'mean_squared_error',
        loss = loss_fn,
        # loss = "categorical_crossentropy",
        metrics = metrics,
        # loss_weights = None,
        # sample_weight_mode = None,
        # weighted_metrics = None,
        # target_tensors = None
    )

    training_saver = data_saver()
    
    training_saver.save_data("act_fun_params", [hidden_layer_1.get_config()['activation']
    , hidden_layer_2.get_config()['activation'], hidden_layer_3.get_config()['activation'], hidden_layer_4.get_config()['activation'], 
    hidden_layer_5.get_config()['activation'], hidden_layer_6.get_config()['activation'], output_layer.get_config()['activation']])

    training_saver.save_data("layers_params", [hidden_layer_1.get_config()['units']
    , hidden_layer_2.get_config()['units'], hidden_layer_3.get_config()['units'], hidden_layer_4.get_config()['units'], 
    hidden_layer_5.get_config()['units'], hidden_layer_6.get_config()['units'], output_layer.get_config()['units']])
    
    # model training  

    return (model, opt, loss_fn, metrics)

def save_model(model, filename="/Users/maciekswiech/Desktop/Praca/B-Droix/Analiza Estetyki CV/models/nn_model.h5"):
    model.save(filename)
    
def model_layers(model):
    return model.layers

def model_inputs(model):
    return model.inputs

def model_outputs(model):
    return model.layers

def model_get_weights(model):
    return model.get_weights

prep_dataset()
