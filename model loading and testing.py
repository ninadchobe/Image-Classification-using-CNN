import numpy as np
from keras.models import model_from_json
import keras.utils as image

with open("model.json","r") as file:
    read_json = file.read()
file.close()

model = model_from_json(read_json)
model.load_weights("model.h5")

def classifly(img):
    img_name = img
    test_img = image.load_img(img_name, target_size=(64, 64))

    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    output = model.predict(test_img)

    if output[0][0] == 0:
        predicted = "F-22"
    else:
        predicted = "Sukhoi Su 30 MKI"
    print(predicted, img)

import os
path = "D:/Data Science/pantech/CNN/Dataset/test"
classes = []
for root, directories, files in os.walk(path):
    for file in files:
        if ".jpeg" in file:
            classes.append(os.path.join(root, file))

for f in classes:
    classifly(f)
    print("\n")