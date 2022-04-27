import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow.io as tfio
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2

from functionality.config import config


def scheduler(epoch):
    return config['learning_rate'] * ((1-config['learning_rate_decay']) ** epoch)


def dataset_from_list(list_paths, label):
    def parse_with_opencv(image_path):
        return cv2.imread(image_path.decode('UTF-8'))

    ds = tf.data.Dataset.from_tensor_slices(list_paths)
    # convert the filenames to actual image tensors
    ds = ds.map(lambda x: tf.numpy_function(parse_with_opencv, [x], Tout=tf.uint8))
    # convert to float and normalize the images to the range [0, 1]
    ds = ds.map(lambda x: tf.cast(x, tf.float32))
    ds = ds.map(lambda x: (x / 255.0))
    # add label to dataset
    ds = ds.map(lambda x: (x, label))
    return ds


def load_dataset(data_dir):
    dataset = None
    counter = 0
    # iterate over all directories in data_dir
    for dir in data_dir.iterdir():
        # get dir name
        dir_name = dir.name
        # print(dir_name)

        # get all files in dir
        dir_files = list(dir.glob("*"))
        dir_paths = [str(file) for file in dir_files]

        # create dataset
        dataset_dir = dataset_from_list(dir_paths, counter)
        # print(len(dataset_dir))

        # we know the dataset structure, so we can use agricultural as our base dataset_dir for concatenation
        if dir_name == 'agricultural':
            dataset = dataset_dir
        else:
            dataset = dataset.concatenate(dataset_dir)
        counter += 1

    # filter out images that are not 256x256
    dataset = dataset.filter(lambda x, y: tf.shape(x)[0] == 256 and tf.shape(x)[1] == 256)

    # shuffle dataset
    dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    # create batches
    dataset = dataset.batch(config['batch_size'])

    return dataset


def print_dataset(dataset):
    for image, label in dataset:
        plt.imshow(image)
        plt.show()
        print(label.numpy().decode('UTF-8'))


def create_model(num_classes):
    return Sequential([
        # create input layer
        layers.InputLayer(input_shape=(256, 256, 3), batch_size=config['batch_size']),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])


def train():
    # load dataset
    train_ds = load_dataset(data_dir=pathlib.Path(config['train_dir']))
    val_ds = load_dataset(data_dir=pathlib.Path(config['val_dir']))

    model = create_model(num_classes=config['num_classes'])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=config['epochs'])

    model.summary()


if __name__ == "__main__":
    train()







