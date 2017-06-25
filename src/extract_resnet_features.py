import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator

import h5py
import os
import time


if __name__=='__main__':
    print('tensorflow version is ', tf.__version__)

    now_path = os.path.abspath(os.path.curdir)
    project_path, tmp = os.path.split(now_path)
    train2_path = os.path.join(project_path, 'train2')
    test_path = os.path.join(project_path, 'test')
    print('train data path is: ', train2_path)

    batch_size = 16
    width = 224
    height = 224
    image_size = [width, height]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    my_resnet = ResNet50(include_top=False, input_tensor=x, weights='imagenet')
    # model = Model(inputs=my_resnet.input, outputs=GlobalAveragePooling2D()(my_resnet.output))



    data_gen = ImageDataGenerator()
    train_data_gen = data_gen.flow_from_directory(train2_path, image_size, shuffle='False', batch_size=batch_size)
    # test_gen =
    # train_gen = my_resnet.predict_generator(train_data_gen, train_data_gen.samples)
    print("train data gen samples is: ", train_data_gen.samples//batch_size)
    print(train_data_gen.batch_size)


    start_time = time.time()
    train_gen = my_resnet.predict_generator(train_data_gen, train_data_gen.samples//batch_size)
    end_time = time.time()
    print('run time of generate feature: ', end_time-start_time)

    with h5py.File('resnet_feature.h5') as h:
        h.create_dataset("trained", data=train_gen)
        h.create_dataset("label", data=train_data_gen.classes)