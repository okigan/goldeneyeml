import coremltools
import cv2

print("starting")

coreml_model = coremltools.utils.load_spec('model.mlmodel')
print(coreml_model)

image = cv2.imread('1-green-empty-bean.png')

print(coreml_model.predict(image))
