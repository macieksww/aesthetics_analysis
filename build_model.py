import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import RandomCrop, RandomFlip, RandomRotation
from keras import initializers 
from keras import regularizers 
from keras import constraints
from keras import losses 
from keras import optimizers 
from keras import metrics 



def build_model(input_dimensions):
    
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
    # input layer
    input_layer = Flatten(input_shape=(input_dimensions[0], input_dimensions[1], 3))
    
    # convolutional layers
    conv_layer_1 = Conv2D(64, kernel_size=3, activation='relu', input_shape=(300,432,3))
    conv_layer_2 = Conv2D(64, kernel_size=3, activation='relu')
    conv_layer_3 = Conv2D(64, kernel_size=3, activation='relu')
    conv_layer_4 = Conv2D(64, kernel_size=3, activation='relu')
    conv_layer_5 = Conv2D(64, kernel_size=3, activation='relu')

    after_conv_layer = Flatten()

    max_pool_2d = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
    
    hidden_layer_1 = Dense(512,  activation = 'linear', kernel_regularizer=l1_regularizer, kernel_initializer = ones_init)
    hidden_layer_2 = Dense(256,  activation = 'relu', kernel_regularizer=l1_regularizer, kernel_initializer = ones_init)
    hidden_layer_3 = Dense(64,  activation = 'relu')
    hidden_layer_4 = Dense(16,  activation = 'relu')
    hidden_layer_5 = Dense(8,  activation = 'relu')
    hidden_layer_6 = Dense(2,  activation = 'relu')
    output_layer = Dense(1, activation = 'sigmoid')

    
    # model definition
    model = Sequential()
    model.add(conv_layer_1)
    model.add(max_pool_2d)
    # model.add(Dropout(0.05)) 
    model.add(conv_layer_2)
    model.add(max_pool_2d)
    # model.add(Dropout(0.05)) 
    model.add(conv_layer_3)
    model.add(max_pool_2d)
    # model.add(Dropout(0.05)) 
    model.add(conv_layer_4)
    model.add(max_pool_2d)
    # model.add(Dropout(0.05)) 
    model.add(conv_layer_5)
    model.add(max_pool_2d)
    # model.add(Dropout(0.05)) 
    # model.add(conv_layer_5)
    # model.add(max_pool_2d)
    # model.add(conv_layer_5)
    # model.add(max_pool_2d)
    model.add(after_conv_layer)
    # model.add(hidden_layer_3)
    # model.add(hidden_layer_4)
    # model.add(hidden_layer_5)
    # model.add(hidden_layer_6)
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
    
    return (model, opt, loss_fn, metrics)
