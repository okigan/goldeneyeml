import cv2
import numpy as np
from keras.models import load_model
import os


def load_and_resize(image_cols, image_rows, path):
    image = cv2.imread(path)
    image = cv2.resize(image, (image_cols, image_rows))
    return image

def load_labels():
    with open('labels.txt') as f:
        lines = f.readlines()
        return {i:label.strip('\n') for i,label in enumerate(lines)}


labels = load_labels()

image_cols, image_rows = 100, 156

model = load_model('./model/weights.h5')

dir = './data/irl-images'
files = os.listdir(dir)
paths = [os.path.join(dir, f) for f in files]

X = np.zeros((len(paths), image_rows, image_cols, 3))

for i, path in enumerate(paths):
    image = load_and_resize(image_cols, image_rows, path)
    X[i, :, :, :] = image.astype('float32') / 255.0

value = model.predict(X)

p = [labels[i] for i in np.argmax(value, 1).tolist()]

print(np.argmax(value, 1))
print(p)
print(files)
