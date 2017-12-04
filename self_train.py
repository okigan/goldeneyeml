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


def main():
    label_images_directory = './data/card-image-trimmed'
    label_image_files = os.listdir(label_images_directory)
    label_image_files.sort()
    label_names = [os.path.splitext(f)[0] for f in label_image_files]
    num_labels = len(label_names)

    image_cols, image_rows, image_channels = 100, 156, 3

    X_train = np.zeros((num_labels, image_rows, image_cols, image_channels))
    y_train = np.zeros((num_labels,))

    for index, filename in enumerate(label_image_files):
        path = os.path.join(label_images_directory, filename)
        X_train[index, :, :, :] = load_and_resize(image_cols, image_rows, path).astype('float32') / 255.0
        y_train[index] = index

    y_train = np_utils.to_categorical(y_train, num_labels)

    model = create_model(num_labels, image_rows, image_cols, image_channels)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    datagen.fit(X_train)

    tensorboard = TensorBoard(log_dir="./logs", write_images=True)

    model_output_dir = './model/'
    os.makedirs(model_output_dir, exist_ok=True)
    mode_basename = 'weights.h5'
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_output_dir, mode_basename), verbose=1)

    sanity_check = SanityCheckCallback(X_train, y_train)

    save_to_dir = None  # './data/datagen'
    if save_to_dir is not None:
        os.makedirs(save_to_dir, exist_ok=True)

    model.fit_generator(datagen.flow(x=X_train, y=y_train, save_to_dir=save_to_dir)
                        , steps_per_epoch=X_train.shape[0]
                        , epochs=2000
                        , callbacks=[tensorboard, model_checkpoint, sanity_check]
                        # , validation_data=(X_train, y_train)
                        )


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


class SanityCheckCallback(keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    # def on_batch_begin(self, batch, logs=None):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X)
        # with K.tf.Session() as sess:
        #     zzz = K.equal(K.argmax(self.y, axis=-1), K.argmax(y_pred, axis=-1))
        #     z = zzz.eval()
        #     pass

        metric = [max(enumerate(m), key=operator.itemgetter(1)) for m in y_pred]

        print("metric:", metric)


def load_and_resize(image_cols, image_rows, path):
    image = cv2.imread(path)
    image = cv2.resize(image, (image_cols, image_rows))
    return image


main()
