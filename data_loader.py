import tensorflow as tf
import os
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def load_data(base_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'valid'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    # print("Class indices:", train_generator.class_indices)
    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    base_dir = 'dataset'
    img_size = (224, 224)
    batch_size = 32
    train_generator, validation_generator, test_generator = load_data(base_dir, img_size, batch_size)
    # for data_batch, labels_batch in train_generator:
    #     print("data batch shape:", data_batch.shape)
    #     print("labels batch shape:", labels_batch.shape)
    #     print("first labels batch:", labels_batch[0])
    #     break
