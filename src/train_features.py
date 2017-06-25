import h5py
import os
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import *


np.random.seed(2017)
batch_size = 128
width = 224
height = 224
image_size = [width, height]


if __name__ == '__main__':
    now_path = os.path.abspath(os.path.curdir)
    print(now_path)

    # 载入数据
    with h5py.File('resnet_feature.h5', 'r') as h:
        x_train = np.array(h['trained'])
        y_train = np.array(h['label'])
    # x_train = np.concatenate(x_train, axis=1)
    # y_train = np.concatenate(y_train, axis=1)
    y_train = y_train[:len(x_train)]
    print(len(y_train), len(x_train))

    x_train, y_train = shuffle(x_train, y_train)

    input_tensor = Input(x_train.shape[1:])
    x = input_tensor
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, x)

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_split=0.2)
