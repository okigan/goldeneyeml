import coremltools

coreml_model = coremltools.converters.keras.convert(
    './model/weights.h5',
    input_names=['input1'],
    image_input_names=['input1'],
    image_scale=1 / 255.,
    class_labels='labels.txt')

coreml_model.save('model.mlmodel')
print(coreml_model)

import cv2

image = cv2.imread('./card-image-trimmed/1-green-empty-bean.png')

print(coreml_model.predict(image))
