from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    test_path = '/home/zhangzhe/pycharm/dogcat/test'
    # [cat, dog]
    # yiyi is 1 if the image is a dog, 0 if cat

    model = load_model('/home/zhangzhe/pycharm/dogcat/src/fine_tuning_dogcat_resnet50')
    print('model load finished')

    files = os.listdir(test_path)
    imgs = []
    for file in files:
        img_path = os.path.join(test_path, file)
        img_pil = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img_pil)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    index = 0
    prediction = []
    file_indexs = []
    for img in imgs:
        result = model.predict(img)
        file_name, suffix = files[index].split('.')
        prediction.append(result[0][1])
        file_indexs.append(file_name)
        index += 1

    df = pd.DataFrame({'id':file_indexs, 'label':prediction})
    df.to_csv('pred.csv', index=None)
    df.head(10)