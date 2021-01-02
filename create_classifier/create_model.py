# Convolutional Neural Networks
# import tensorflow as tf
# from tensorflow import keras
import numpy as np

# Importing Keras libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array


def create_model():
    model = Sequential()

    # Step 1 - Convolution
    input_shape = (64, 64, 3)  # 3 channels (coloured image) and image dimensions
    model.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

    # Step 2 - Pooling (Max pooling technique)
    # # We are reducing the complexity keeping information
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding second Convolution layer
    model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    # # Put the pooled feature map in one big vector
    model.add(Flatten())

    # Step 4 - Full connection
    model.add(Dense(128, activation='relu'))
    # Binary outcome -> sigmoid
    # More -> softmax
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the CNN
    # # Binary outcome -> binary_crossentropy
    # # More -> categorical_crossentropy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the CNN to the image
    # # We are using image augmentation
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory(
            'dataset/training_set',
            target_size=(64, 64), # size of images
            batch_size=32,
            class_mode='binary')
    test_set = test_datagen.flow_from_directory(
            'dataset/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')
    model.fit(
            training_set,
            steps_per_epoch=8000//32,  # Numbers of image
            epochs=25,
            batch_size=32,
            validation_data=test_set,
            validation_steps=2000//32)

    model.save("model.h5")
    return model
