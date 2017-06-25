### 文件说明
- mklink_category.py

创建train2文件夹，将猫狗图片的link分别存在cat/dog文件夹下。

- mklink_trainval.py

创建train3和val3文件夹，将原始训练集分为训练集和测试集两部分。

- extract_resnet_features.py

提取ResNet50的bottleneck特征，并保存到h5文件。

- train_features.py

读取通过ResNet50提取的特征，训练一个简易分类网络。

- transfer_learning.py

基于迁移学习，fine-tuning修改的ResNet50，并保存网络。

- load_model.py	

读取网络，应用于测试集，并将结果保存为kaggle要求的格式。


