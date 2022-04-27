import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import PIL.Image as Image
from  tensorflow.keras.applications.resnet50 import preprocess_input

from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# define callback functions
def scheduler(epoch):
    return BASE_LR * (1-LR_REDUCTION)**epoch

# dataset paths
train_dir = "data/Challenge_dataset/train"
validation_dir = "data/Challenge_dataset/validation"
test_dir = "data/Challenge_dataset/test"

# hyper-parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
BASE_LR = 0.002
LR_REDUCTION = 0.08
EPOCHS = 20



# load datasets via image data generator, that rescales the images and preprocesses them with the function of resnet50
generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
train_dataset = generator.flow_from_directory(train_dir, batch_size=BATCH_SIZE, target_size=IMG_SIZE)
validation_dataset = generator.flow_from_directory(validation_dir, batch_size=BATCH_SIZE, target_size=IMG_SIZE)
test_dataset = generator.flow_from_directory(test_dir, batch_size=BATCH_SIZE, target_size=IMG_SIZE)

# download resnet model pretrained on big-earth dataset and use it as general feature extractor
base_model_link = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"
base_model_layer = hub.KerasLayer(base_model_link, input_shape=IMG_SHAPE, trainable=False)
base_model = tf.keras.Sequential([base_model_layer])

# create a data augmentation layer for improving data generalization
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# add classification head: a single dense layer to output the prediction for all 21 classes
prediction_layer = tf.keras.layers.Dense(21)

# create model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy', f1_m, precision_m, recall_m])

# train the model
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[callback])

