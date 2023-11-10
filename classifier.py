import cv2
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# set seed
tf.random.set_seed(0)

img_path = cv2.imread("./assets/d1.jpg")
img_path = cv2.resize(img_path, (224, 224))
# cv2.imshow("img_path", img_path)
# cv2.waitKey(0)

model = keras.Sequential()

# Block 1
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape =(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block2
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block3
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block4
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

# Block5
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

model.build()
model.summary()

# Result

result = model.predict(np.array([img_path]))

for i in range(64):
    feature_img = result[0, :, :, i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")
plt.show()