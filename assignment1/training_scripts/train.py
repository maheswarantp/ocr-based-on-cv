import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def init_datagen(df, batch_size = 32, target_size = (256, 256)):
    datagen = ImageDataGenerator(
        rescale = 1./255,
        validation_split = 0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    # Define training and validation data generators
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_paths',
        y_col='image_labels',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Use 'sparse' for integer labels
        subset='training'
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_paths',
        y_col='image_labels',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Use 'sparse' for integer labels
        subset='validation'
    )

    return datagen, train_generator, validation_generator