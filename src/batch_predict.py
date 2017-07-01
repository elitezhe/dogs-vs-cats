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

    prediction = []
    file_indexs = []
    im_ind = 0
    print('Start Prediction: ')
    for file in files:
        print(im_ind, file)
        img_path = os.path.join(test_path, file)
        img_pil = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img_pil)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img)
        file_name, suffix = file.split('.')
        prediction.append(result[0][1])
        file_indexs.append(file_name)
        print('     ', result[0][1], file_name)
        im_ind += 1
    print('Finish prediction. ')

    df = pd.DataFrame({'id':file_indexs, 'label':prediction})
    df.to_csv('pred2.csv', index=None)
    df.head(10)
