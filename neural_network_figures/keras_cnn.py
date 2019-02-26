from __future__ import print_function
import os
import matplotlib.image as mpimg
import numpy as np  # linear algebra
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt


# setup the NN structure
PATH = os.path.abspath(os.path.curdir)
SHAPES = {
    0: 'triangles',
    1: 'circles',
    2: 'squares'
}

batch_size = 128
num_classes = 3
epochs = 10

# input image dimensions
img_x, img_y = 28, 28


def rgb2gray(img):
    img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return 1 - img


# load data and scale
# digits = load_digits()
files = []  # for storing all the images path
result = []
for index, shape in SHAPES.items():  # |
    new_path = os.path.join(PATH, shape)  # |
    for file_ in os.listdir(new_path):  # |How can I make this code shorter?
        files.append(os.path.join(new_path, file_))  # |
        result.append(index)

images = []  # list for images
for file_ in files:
    img = mpimg.imread(file_)
    img = rgb2gray(img)
    img = np.ravel(img)
    images.append(img)

X_scale = StandardScaler()
# X = X_scale.fit_transform(digits.data)
# y = digits.target

X = X_scale.fit_transform(images)
y = result
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(40, kernel_size=(9, 9), strides=(1, 1),
                 activation='tanh',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(880, (3, 3), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='tanh'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
