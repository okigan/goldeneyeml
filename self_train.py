import operator
import os

import cv2
import keras
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


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
        X_train[index, :, :, :] = load_and_resize(image_cols, image_rows, path).astype('float32')
        y_train[index] = index

    y_train = np_utils.to_categorical(y_train, num_labels)

    model_output_dir = './model/'
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    mode_basename = 'weights.h5'
    model_filepath = os.path.join(model_output_dir, mode_basename)

    if os.path.exists(model_filepath):
        model = load_model(model_filepath)
    else:
        model = create_model(num_labels, image_rows, image_cols, image_channels)
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        channel_shift_range=64,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0 / 255.0
    )
    datagen.fit(X_train)

    datagen_test = ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    datagen_test.fit(X_train)

    tensorboard = TensorBoard(log_dir="./logs", write_images=True)

    model_checkpoint = ModelCheckpoint(filepath=model_filepath, save_best_only=True, verbose=1)

    sanity_check = SanityCheckCallback(X_train, y_train)

    save_to_dir = None  # './data/datagen'
    save_to_dir = './data/datagen'
    if save_to_dir is not None and not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    model.fit_generator(datagen.flow(x=X_train, y=y_train, save_to_dir=save_to_dir)
                        , steps_per_epoch=X_train.shape[0]
                        , epochs=2000
                        , callbacks=[tensorboard, model_checkpoint, sanity_check]
                        , validation_data=datagen_test.flow(x=X_train, y=y_train)
                        , validation_steps=20
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
        super(SanityCheckCallback, self).__init__()
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


print("image_dim_ordering:", keras.backend.image_dim_ordering())
print("image_data_format:", keras.backend.image_data_format())

main()
