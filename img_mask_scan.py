import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import math

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open('test_photo1.jpg')

size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)
okay = round(prediction[0][0] * 100, 1)
okay2 = "마스크 착용 : " + str(okay) + "%"
noo = round(prediction[0][1] * 100, 1)
noo2 = "마스크 미착용 : " + str(noo) + "%"
# print(prediction)
print(okay2)
print(noo2)
txtp = ["===> 마스크 착용으로 예상됨 <===", "===> 마스크 미착용으로 예상됨 <==="]
print(txtp[np.argmax(prediction[0])])
