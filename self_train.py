import operator
import os

import cv2
import keras
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

print("image_dim_ordering:", keras.backend.image_dim_ordering())
print("image_data_format:", keras.backend.image_data_format())


def create_model(num_classes, image_rows, image_cols, image_channels):
    model = Sequential()

    model.add(Convolution2D(32, 3, input_shape=(image_rows, image_cols, image_channels,)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


label_images_directory = './data/card-image-trimmed'
label_image_files = os.listdir(label_images_directory)
label_image_files.sort()
label_names = [os.path.splitext(f)[0] for f in label_image_files]
num_classes = len(label_names)

image_cols, image_rows, image_channels = 100, 156, 3

X = np.zeros((num_classes, image_rows, image_cols, image_channels))
y = np.zeros((num_classes,))

for index, filename in enumerate(label_image_files):
    path = os.path.join(label_images_directory, filename)
    image = cv2.imread(path)
    image = cv2.resize(image, (image_cols, image_rows))
    X[index, :, :, :] = image
    y[index] = index

y = np_utils.to_categorical(y, num_classes)

model = create_model(num_classes, image_rows, image_cols, image_channels)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last"
)

tensorboard = TensorBoard(log_dir="./logs", write_images=True)

model_output_dir = './model/'
os.makedirs(model_output_dir, exist_ok=True)
mode_basename = 'weights.h5'
model_filepath = os.path.join(model_output_dir, mode_basename)
model_checkpoint = ModelCheckpoint(filepath=model_filepath, verbose=1)

datagen.fit(X)


class My_Callback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        value = model.predict(data)
        print("prediction", max(enumerate(value[0]), key=operator.itemgetter(1)))
        value = model.predict(data)
        datagen.random_transform(X[0])
        return


input_path = './data/card-image-trimmed/3-red-striped-oval.png'
image = cv2.imread(input_path)
image = cv2.resize(image, (image_cols, image_rows))
data = image.reshape(1, image_rows, image_cols, image_channels)
value = model.predict(data.astype('float32') / 255)

my_callback = My_Callback(data)

model.fit_generator(datagen.flow(X, y)
                    , steps_per_epoch=num_classes
                    , epochs=2000
                    , callbacks=[tensorboard, model_checkpoint, my_callback]
                    , validation_data=(X, y)
                    )
