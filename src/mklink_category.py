import shutil
import os


def rmrf_mkdir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
        print('delete: ', dirpath)
    print('make dir: ', dirpath)
    os.mkdir(dirpath)


if __name__ == '__main__':
    now_path = os.path.abspath(os.path.curdir)
    print(now_path)
    project_path, tmp = os.path.split(now_path)
    # print(project_path, tmp)
    train_path = os.path.join(project_path, 'train')
    train2_path = os.path.join(project_path, 'train2')
    print('train data path is: ', train_path)

    rmrf_mkdir(train2_path)
    os.mkdir(os.path.join(train2_path, 'dog'))
    os.mkdir(os.path.join(train2_path, 'cat'))

    train_filenames = os.listdir(train_path)
    print('total # of file :', len(train_filenames))
    train_cat = filter(lambda x: x[:3]=='cat', train_filenames)
    train_dog = filter(lambda x: x[:3] =='dog', train_filenames)
    # print('# of cat pictures: ', len(train_cat))
    # print('# of dog pictures: ', len(train_dog))

    train2_dog_path = os.path.join(train2_path, 'dog')
    train2_cat_path = os.path.join(train2_path, 'cat')
    for filename in train_cat:
        os.symlink(os.path.join(train_path, filename), os.path.join(train2_cat_path, filename))
    for filename in train_dog:
        os.symlink(os.path.join(train_path, filename), os.path.join(train2_dog_path, filename))
