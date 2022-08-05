import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def data_augmentation(data, type):
    flip_layer = tf.keras.layers.RandomFlip('horizontal')
    rotation_layer = tf.keras.layers.RandomRotation(0.2)
    full_augmentation_layer = keras.Sequential(
    [
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ]
    )

    rotated_ds = data.map(lambda x, y: (rotation_layer(x), y))
    flipped_ds = data.map(lambda x, y: (flip_layer(x), y))
    fully_augmented_ds = data.map(lambda x, y: (full_augmentation_layer(x), y))

    if type is 'roatation':
        return rotated_ds
    elif type is 'flip':
        return flipped_ds
    elif type is 'full':
        return fully_augmented_ds
    else:
        return flipped_ds
