import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import return_model
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


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

def init_training(df):
    model = return_model(input_shape=(256, 256), num_classes=8, is_training=True)
    datagen, train_generator, validation_generator = init_datagen(df, batch_size=32, target_size=(256, 256))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
    filepath="top_model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stopping,checkpoint]
    
    history = model.fit(train_generator, validation_data=validation_generator, epochs=5, callbacks = callbacks_list)

    return model, history
