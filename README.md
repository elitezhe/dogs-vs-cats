## Dogs VS Cats
### 0x01 介绍
本项目来源于Kaggle上的一个竞赛：Dogs vs. Cats Redux: Kernels Edition[^1]，该竞赛的主要任务就是分辨猫和狗的图片。
在编写该程序的过程中还参考了ypwhs/dogs_vs_cats[^3]的项目。
我的项目地址：https://github.com/elitezhe/dogs-vs-cats


### 0x02 数据
项目提供的数据包含训练集和测试集，训练集共包含25,000张图片，其中猫狗各一半，文件名以cat或dog开头表明该图片的label。测试集无标注（废话。。），只能导出测试结果为项目要求的标准格式，上传到测试服务器，直接给出logloss来评估自己的结果。（logloss越小，结果越好）

### 0x03 模型
我采用了resnet50作为基本模型，移除了最后一层1000元素的softmax，用了一层2元素的softmax代替;另一种更简单的方案可以直接用1个元素的logistic代替，并且该方法在最后导出结果时更方便。
设置要训练的层。通常迁移学习保留低层，训练高层，具体可以参考cs231n的一篇post[^2]。
```
for layer in model.layers[140:]:
        layer.trainable = True
```
图片数据的预处理采用了keras提供的```ImageDataGenerator()```，可以在训练的同时从磁盘读取下一个batch，无需一次性把所有图片全部载入内存；并且可以直接resize图片。不过其中具体机理，我还没有弄得十分明白，比如resnet文章中提到的per-pixel mean，究竟有没有，我也不清楚（摊手.jpg），不过从最终结果来看，这里并没有影响精度。

### 0x04 结果
跑了5个epchos，最终测试集上正确率好像是超过99%了（记不太清楚了，真该用jupyter notebook的，就可以直接看到当时的运行结果了。）
最终我的成绩0.19433（logloss）,在kaggle公布的排名上在700-800名(由于竞赛已经结束，所以后续提交的不会再公布在该列表上)


### 参考
[^1]: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

[^2]: http://cs231n.github.io/transfer-learning/

[^3]: https://github.com/ypwhs/dogs_vs_cats
