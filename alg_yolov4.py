from sys import path_hooks
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from import_cv import process_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from keras.models import Sequential 
from keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from save_train_data import data_saver
from data_augmentation import data_augmentation
from build_efficientnet_model import build_model
from training_summary import save_model, model_inputs, model_layers, model_outputs, model_get_weights, plot_images, training_summary
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

epochs = 15
batch_size = 32
load_model = False
early_stopping = False
print("BATCH SIZE")
print(batch_size)
image_size = (300, 432)
efficientnet_density = 4
data_dir = "/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432/"
aesthetic_dir = "/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432/aesthetic/"
nonaesthetic_dir = "/home/bdroix/bdroix/aesthetics_analysis/dane do analizy/dane_scaled_300_432/nonaesthetic/"

def train():
    global epochs
    global batch_size
    global load_model
    global early_stopping
    global image_size
    global data_dir
    global aesthetic_dir
    global nonaesthetic_dir
    global efficientnet_density

    # number of all images in dataset
    num_of_images = process_directory(data_dir)
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


    
    # preparation of filenames to later set predictions with validation files
    print("FILENAMES")
    file_paths = val_ds.file_paths
    temp_file_paths = []
    for file_path in file_paths:
        if "non" in file_path:
            file_path = file_path[89:]
        else:
            file_path = file_path[86:]
        file_path = np.array([file_path])
        temp_file_paths.append(file_path)
    file_paths = np.array(temp_file_paths)

    class_names = train_ds.class_names
    print("DATA CLASSES:")
    print(class_names)

    # data augmentation
    augmented_train_ds = data_augmentation(train_ds, "flip")
    augmented_val_ds = data_augmentation(val_ds, "flip")

    # print(train_ds.__len__())
    # print(train_ds.cardinality().numpy())

    # concatenation of training and validation dataset
    train_ds.concatenate(augmented_train_ds)
    val_ds.concatenate(augmented_val_ds)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    train_ds = normalized_train_ds
    normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = normalized_val_ds
    image_batch, labels_batch = next(iter(normalized_train_ds))
    first_image = image_batch[0]
    
    # prefetch and caching for better dataset performace
    # caching keeps images in caceh after they are loaded
    # from disk during first epoch
    # prefetch overlaps data preprocessing and model execution

    # print("TRAIN DATASET ELEMENTS")
    # for element in train_ds.as_numpy_iterator(): 
    #     print(element[0])
    #     print(type(element[0]))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    if load_model is False:
        model = build_model([image_size[0], image_size[1]], efficientnet_density)[0]
    else:
        model = load_model('model.h5')

    opt = build_model([image_size[0], image_size[1]], efficientnet_density)[1]
    loss_fn = build_model([image_size[0], image_size[1]], efficientnet_density)[2]
    metrics = build_model([image_size[0], image_size[1]], efficientnet_density)[3]

    csv_logger = CSVLogger("/home/bdroix/bdroix/aesthetics_analysis/model_history_log_adagrad_32.csv", append=False)
    # model.summary()

    # stopping after reaching certain accuracy instead of constant num of epochs
    es = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1)

    if early_stopping is False:
        history = model.fit(train_ds, batch_size = batch_size, validation_data = val_ds, epochs=epochs, callbacks=[csv_logger])
    else:
        history = model.fit(train_ds, batch_size = batch_size, validation_data = val_ds, epochs=epochs, callbacks=[csv_logger, es])
    
    # training summary
    training_summary(model, val_ds, file_paths, opt, loss_fn, metrics, epochs, batch_size)
    

train()
