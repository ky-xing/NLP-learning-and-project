# 文本分类—RNN and CNN

为了更好地熟悉Pytorch、CNN和RNN，本项目采用Pytroch，使用卷积神经网络和循环神经网络来进行文本分类

## 环境

* python==3.7.1
* pytorch==1.3.1
* tensorflow==1.12
* numpy
* pandas
* sklearn

## 数据集

使用THUCNews新闻数据集的一个子集，完整数据集在http://thuctc.thunlp.org/下载，请按照开源协议使用

子数据集，包括5w条训练集，5000条验证集，10000条测试集，共10个类别

类别主要有：**体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐**

子集下载地址：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

 为了提高训练效率，并未采用全量数据来进行训练，对训练集进行采样，每种新闻类别采样1000条，共1w条数据来作为训练样本，采样过程见train_sampling.py

## 数据预处理

data_loader.py包含多个处理函数：

* `read_vocab()` :读取已经生成的词表，并转换为`{词：index}`的表示，此处所采用的词表为字符级别的；
* `read_category()`: 将分类目录固定，转换为`{类别: index}`表示；
* `process_file()`:读取文本，并将文本和标签转换为字词索引，采用tf.kears.pad_sequence将句子长度进行统一，长度统一为600；
* `data_load()`:按照batch_size来生成每一个batch的x_train和y_train

处理后的数据格式：

x_train.shape : [10000，600]    y_train.shape : [10000,10]

## CNN、RNN的模型

RNN的模型结构在model.py中，主要包括embedding层、两层LSTM层、两层全连接层，代码如下：

```
class TextRnn(nn.Module):
  def __init__(self):
​    super(TextRnn,self).__init__()
​    self.embedding = nn.Embedding(5000,64)
​    self.rnn = nn.LSTM(input_size = 64,hidden_size = 128,num_layers=2,bidirectional = True）
​    self.f1 = nn.Sequential(nn.Linear(256,128),
​                nn.Dropout(0.2),
​                nn.ReLU())
​    self.f2 = nn.Sequential(nn.Linear(128,10),
​                nn.Softmax())
  def forward(self,x):
​    x = self.embedding(x)
​    x,_ = self.rnn(x)
​    x = f.dropout(x,0.5)
​    x = self.f1(x[:,-1,:])
​    return self.f2(x)
```

CNN的模型结构与rnn类似，只不过将LSTM层换为卷积层，并只采用一层全连接层，代码如下：

```
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=64,
                                        out_channels=256,
                                        kernel_size=5),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=596))
        self.f1 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.embedding(x) # batch_size x* text_len x *embedding_size 64*600*64
        x = x.permute(0, 2, 1) #64*64*600
        x = self.conv(x)  #Conv1后64*256*596,ReLU后不变,NaxPool1d后64*256*1
        x = x.view(-1, x.size(1)) #64*256
        x = f.dropout(x,0.5)
        x = self.f1(x)    #64*10 batch_size * class_num
        return x
```

主要调整的超参数：

```
embedding_dim = 64      # 词向量维度      
vocab_size = 5000       # 词汇表达小     
num_layers= 2           # 隐藏层层数    
hidden_dim = 128        # 隐藏层神经元    
rnn = 'gru'             # lstm 或 gru     
dropout_keep_prob = 0.5/0.8 # dropout保留比例    
learning_rate = 1e-3    # 学习率     
batch_size = 32/64      # 每批训练大小    
num_epochs = 10/50      # 总迭代轮次     
```

## 训练结果

CNN迭代8轮，验证集准确率达到0.96，训练速度和收敛速度均比RNN快。

RNN采用LSTM层，训练速度较慢，迭代28轮次，验证集准确率只有0.89，收敛速度相比CNN慢了很多。

由于本次训练数据生成的是字符级的文本，CNN网络的表现比RNN要好，RNN要获得较好的结果需要更长的训练时间。

另外，还可以进一步调整参数，以获得更好的结果。
## 参考资料
1.[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)

2.[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

3.[理解LSTM网络](https://blog.csdn.net/juanjuan1314/article/details/52020607)
