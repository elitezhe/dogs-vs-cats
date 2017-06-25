from keras.models import *
from keras.layers import *
from keras.applications import ResNet50
from keras.preprocessing.image import *
from keras.optimizers import *

import numpy as np
import os

np.random.seed(2017)

from skimage.io import imread_collection, imread
from skimage.transform import resize


if __name__ == '__main__':

    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg',
                          input_tensor=Input((224, 224, 3)))
    for layer in base_model.layers:
        layer.trainable = False
    # for tmp in zip([x.name for x in base_model.layers], range(len(base_model.layers))):
    #     print(tmp)

    # x = Flatten()(base_model.output)
    x = Dense(2, activation='softmax', name='fc2')(base_model.output)

    model = Model(base_model.input, x)

    # for tmp in zip([x.name for x in model.layers], range(len(model.layers))):
    #     print(tmp)

    for layer in model.layers[140:]:
        layer.trainable = True

    # load data
    # now_path = os.path.abspath(os.path.curdir)
    # project_path, tmp = os.path.split(now_path)
    # train_path = os.path.join(project_path, 'train')
    train_path = '/home/zhangzhe/pycharm/dogcat/train'
    train3_path = '/home/zhangzhe/pycharm/dogcat/train3'
    val3_path = '/home/zhangzhe/pycharm/dogcat/val3'

    n = 25000
    # X = np.zeros((n, 224, 224, 3), dtype=np.uint8)
    # y = np.zeros((n, 2), dtype=np.float32)
    #
    # for i in range(n//2):
    #     X[i] = resize(imread(os.path.join(train_path, 'cat.%d.jpg' % i)), (224, 224, 3))
    #     X[i+n//2] = resize(imread(os.path.join(train_path, 'dog.%d.jpg' % i)), (224, 224, 3))
    #     if i % 1000 == 0:
    #         print(i)
    #
    # y[:n//2] = np.array([1.0, 0.0])
    # y[n//2:] = np.array([0.0, 1.0])
    #
    # # optiz = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    optiz = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #
    model.compile(optimizer=optiz,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # # X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.1)
    # model.fit(X, y, batch_size=32, epochs=15, validation_split=0.1)

    data_gen = ImageDataGenerator()
    train_data_gen = data_gen.flow_from_directory(train3_path, (224, 224), shuffle='False', batch_size=32)
    val_data_gen = data_gen.flow_from_directory(val3_path, (224, 224), shuffle='False', batch_size=32)

    model.fit_generator(train_data_gen, 1000, 5, validation_data=val_data_gen, validation_steps=10)

    model.save('fine_tuning_dogcat_resnet50')