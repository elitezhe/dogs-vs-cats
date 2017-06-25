import shutil
import os
from sklearn.model_selection import train_test_split


def rmrf_mkdir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
        print('delete: ', dirpath)
    print('make dir: ', dirpath)
    os.mkdir(dirpath)


if __name__ == '__main__':

    train_val_split = 0.90

    now_path = os.path.abspath(os.path.curdir)
    print(now_path)
    project_path, tmp = os.path.split(now_path)
    # print(project_path, tmp)
    train_path = os.path.join(project_path, 'train')
    train3_path = os.path.join(project_path, 'train3')
    val3_path = os.path.join(project_path, 'val3')
    print('train data path is: ', train_path)

    rmrf_mkdir(train3_path)
    rmrf_mkdir(val3_path)
    os.mkdir(os.path.join(train3_path, 'dog'))
    os.mkdir(os.path.join(train3_path, 'cat'))
    os.mkdir(os.path.join(val3_path, 'dog'))
    os.mkdir(os.path.join(val3_path, 'cat'))

    train_filenames = os.listdir(train_path)
    train_files, val_files = train_test_split(train_filenames, test_size=(1-train_val_split))
    print('total # of file :', len(train_filenames))
    train_cat = filter(lambda x: x[:3]=='cat', train_files)
    train_dog = filter(lambda x: x[:3] =='dog', train_files)
    # print('# of cat pictures: ', len(train_cat))
    # print('# of dog pictures: ', len(train_dog))

    train3_dog_path = os.path.join(train3_path, 'dog')
    train3_cat_path = os.path.join(train3_path, 'cat')
    for filename in train_cat:
        os.symlink(os.path.join(train_path, filename), os.path.join(train3_cat_path, filename))
    for filename in train_dog:
        os.symlink(os.path.join(train_path, filename), os.path.join(train3_dog_path, filename))

    val3_dog_path = os.path.join(val3_path, 'dog')
    val3_cat_path = os.path.join(val3_path, 'cat')
    val_cat = filter(lambda x: x[:3] == 'cat', val_files)
    val_dog = filter(lambda x: x[:3] == 'dog', val_files)
    for filename in val_cat:
        os.symlink(os.path.join(train_path, filename), os.path.join(val3_cat_path, filename))
    for filename in val_dog:
        os.symlink(os.path.join(train_path, filename), os.path.join(val3_dog_path, filename))