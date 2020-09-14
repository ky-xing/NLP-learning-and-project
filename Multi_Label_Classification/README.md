# Multi-Label-Text-classification

 本项目是商品的细粒度用户评论分析，属于多标签分类任务。

项目采用的数据集为电商平台的用户对商品的评价信息数据集，按照商品种类、用户群体、评价情绪、评价标签等多种标签构成，商品种类、用户评价情绪有多种，属于多分类，可构建多分类任务；而在评价标签在文本当中出现，需要在数据预处理的时候，将评价标签提取出来。

数据提取后，评价标签总共有900多个，并且每个评论信息可能有多个标签，属于多标签分类，该任务可以通过对输出层和损失函数进行改造，转换为N个二分类任务，其中N为标签个数。

针对多标签文本分类，本项目采用TFIDF+机器学习的baseline模型，以及TextCNN和Transformer的深度学习模型，其中TextCNN模型在训练过程种使用了Word2vec或Fasttext的预训练词向量

本项目为学习项目，参考一些他人的博客和代码，通过代码实践与理论相结合，加深对相关文本分类模型的理解和认识

## 1.项目环境

- Python==3.7.1
- scikit-learn==0.21.2
- pandas==0.24.2
- numpy==1.18.2
- tensorflow==2.0.0

## 2.项目框架

项目大致包括以下部分：

```python
│─data
    │─高中.zip  #数据源
    │─hit_stopwords.txt  #哈工大停用词表
	|-vocab.txt #词表
	|-x_95.npy	#处理后的文本索引序列
	|-y_95.npy	#处理后的标签，此为95分类标签，可自行处理数据
│─data_load.py   #整合数据，过滤掉标签出现次数较少的样本
|-data_precess.py  #预训练词向量，构建词表，并转换文本序列
|-baseline   #传统的机器学习算法
│─textcnn  # 模型的搭建和训练代码，下同
	|model.py
	|train.py
	|test.py
│─transformer 
	|layer.py
	|model.py
	|train.py
	|test.py
```

1、数据预处理

* data_load.py  对多个csv数据表进行整合，根据文件名字附加标签，并从文本内容中提取商品评论标签，之后对数据进行过滤，去掉标签出现次数少于300的标签和样本
* data_precess.py  对数据进行清洗，去除数字、符号、冗余字符等，用jieba分词后去除停用词，使用word2vec和fasttext来训练词向量，建立词表，

2、模型搭建

**1）baseline模型**

采用TF-IDF获得文本特征向量，采用Naive-Bayes、LR、RandomForest等机器学习模型，分别进行4分类和95标签分类，四分类准确率可达0.9，95标签分类micro-f1  0.46，多标签分类效果较差

**2）TextCNN模型**

模型的相关原理可以看[TextCNN原理](https://blog.csdn.net/pipisorry/article/details/85076712)

模型的输入`x`是经xpadding处理过后的文本索引序列，`x_max_len = 150`，数据处理后共95个标签，输出`y`为95维one-hot形式的矩阵，输出层采用的是`sigmoid`激活函数，即将多标签分类变为95个二分类任务

4分类任务的输出采用`Softmax`激活函数，损失函数为`loss = 'categorical_crossentropy'`

95多标签分类采用的损失函数`loss = 'binary_crossentropy'`

采用word2vec预训练词向量时，将embdding层的weights设置为`word2vec`得到的`embedding_matrixs`

另外还可以将词向量随模型的训练进行更新，只需将`tf.keras.layers.embedding(trainable=True)`，这样可以达到fine tuning的效果，通过对比发现`trainable=True`在同样训练参数下，可以使模型micro-f1提升5~8%。

迭代20个epoch，测试集micro-f1 0.80  macro-f1 0.49， 由于样本的标签存在不均衡的问题，macro-f1得分并不高，

针对模型的结果，通过调整部分参数来提高模型得分，调整的主要参数有：

kernel_size(2~5)  、embedding_size(128/200/300) 、 filter_num(2/3/4)、dropout_rate等 

**3）Transformer**

Transformer在tensorflow官网有[详细教程](https://tensorflow.google.cn/tutorials/text/transformer)，我结合一些博客和相关代码也进行一些学习和使用

Transformer通过引入self-Attention的机制，捕捉丰富的多维特征以及语义信息，并且不会随序列长度增加而产生信息丢失。

针对本项目的文本分类任务，只使用了Transformer的Encoder层，之后接全连接层并使用sigmoid激活函数，直接得到分类结果的概率

为加速训练速度(无GPU可用)，Encoder层只使用了4层，8头attention，具体参数见Transformer_train.py。

经测试，Transformer性能比TextCNN好很多，迭代5个epoch，验证集micro-f1 0.86  macro-f1 0.65，已完全超过TextCNN的性能，迭代10个epoch，验证集micro_f1 0.9093 macro_f1 0.7474，测试集micro_f1 0.8936 macro_f1 0.7862

虽然Transformer性能更好，但参数多且随层数增加而增加，训练时间周期较长。

## 3.模型结果评估

4分类任务采用precison，recall，acc来进行评估，TextCNN表现较为出色，分类准确率接近0.99

多标签分类采用micro f1 和macro f1来进行评价

micro-f1 是计算出所有类别总的Precision和Recall，然后计算F1 

macro-f1 是 计算出每一个类的Precison和Recall后计算F1，最后将F1平均 

本项目自定义了两个评价指标，具体见metrics.py

以下为各个模型在测试集的表现：

| 模型                     | micro f1 | macro f1 | precision | recall | acc   |   备注   |
| ------------------------ | -------- | -------- | --------- | :----: | ----- | :------: |
| baseline(tfidf+ml)-4     |          |          | 0.88      |  0.89  | 0.89  |          |
| TextCNN-4                |          |          | 0.99      |  0.98  | 0.989 | 10，1e-3 |
| baseline(tfidf+ml)-95    | 0.46     | 0.11     | 0.59      |  0.37  |       |          |
| TextCNN-95-without w2vec | 0.6703   | 0.2491   | 0.86      |  0.55  |       | 20，1e-3 |
| TextCNN-95-w2vec不更新   | 0.7652   | 0.3534   | 0.87      |  0.42  |       | 20，1e-3 |
| TextCNN-95-w2vec更新     | 0.8032   | 0.4832   | 0.8912    | 0.7316 |       | 20，1e-3 |
| Transformer-95           | 0.8858   | 0.8200   | 0.8860    | 0.8857 |       | 10，1e-3 |

## 总结

TextCNN关注文本的局部特征，对于文本分类任务具有很好的表现，同时速度也很快，使用word2vec预训练并进行fine-tuning，可以提高模型的分类效果

Transformer在信息捕捉方面具有更好的性能，模型的表现相比TextCNN有很不错的提升，若使用更深层的Encoder将会有更好的表现。

### 参考资料

1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
2. [TextCNN原理](https://blog.csdn.net/pipisorry/article/details/85076712)
3. [评价指标](https://blog.csdn.net/sinat_28576553/article/details/80258619)
4. [Attention Is All You Need]( https://arxiv.org/abs/1706.03762 )
5. [Transformer文本分类实战](https://zhuanlan.zhihu.com/p/105036982?utm_source=cn.wiz.note)
6. [Text-Classification](https://github.com/Light2077/Text-Classification) 

