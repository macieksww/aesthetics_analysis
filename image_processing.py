import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



  # model.fit(
    #         train_generator,
    #         steps_per_epoch=2000,
    #         epochs=50,
    #         validation_data=validation_generator,
    #         validation_steps=800)

        
# def make_model(input_shape, num_classes):
#     inputs = keras.Input(shape=input_shape)
#     # Image augmentation block
    
#     # No use of data augmentation. Random transformations 
#     # would destroy the original composition of CV which is
#     # a crucial factor is deciding wether or not it is aesthetic
    
#     # x = data_augmentation(inputs)
#     x = inputs

#     # Entry block
#     # rescaling of rgb values of pixels 
#     # from (0-255) to (0-1) which is natural space for NN
    
#     x = layers.Rescaling(1.0 / 255)(x)
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     x = layers.Conv2D(64, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     previous_block_activation = x  # Set aside residual

#     for size in [128, 256, 512, 728]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(size, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)

#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

#         # Project residual
#         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual

#     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)

#     x = layers.GlobalAveragePooling2D()(x)
#     if num_classes == 2:
#         activation = "sigmoid"
#         units = 1
#     else:
#         activation = "softmax"
#         units = num_classes

#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(units, activation=activation)(x)
#     return keras.Model(inputs, outputs)


# model = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)


    # plot of dataset
    
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")
    # plt.show()
    
    
        # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     shear_range=0.0,
    #     zoom_range=0.0,
    #     horizontal_flip=False,
    #     rotation_range=0,
    #     width_shift_range=0.0,
    #     height_shift_range=0.0,
    #     vertical_flip=False,
    # )
    
    # test_datagen = ImageDataGenerator(rescale=1./255)
    
    # train_generator = train_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=(image_size[0], image_size[1]), # size of input imgs
    #         batch_size=batch_size, # no of imgs yielded to generator per batch
    #         class_mode='binary', # type of categories
    #         shuffle=False, # shuffilng of order of imgs yielded to the generator
    #         seed=42, # random seed for applying random data shuffling and augmentation
    #         color_mode='grayscale',
    #     )
        
    # validation_generator = test_datagen.flow_from_directory(
    #         validation_dir,
    #         target_size=(image_size[0], image_size[1]), # size of input imgs
    #         batch_size=batch_size, # no of imgs yielded to generator per batch
    #         class_mode='binary',
    #         shuffle=False, # type of categories
    #         seed=42, # random seed for applying random data shuffling and augmentation
    #         color_mode='grayscale',
    #     )
    
