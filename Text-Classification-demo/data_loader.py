import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs

def read_vocab(file_path):
    with open(file_path,encoding='utf-8',errors='ignore') as f:
        words = [line.strip() for line in f.readlines()]
    word_to_id = dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    categorys = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categorys = [x for x in categorys]
    cat_to_id = dict(zip(categorys,range(len(categorys))))
    return categorys,cat_to_id


#读取文本，并将其转换为字词索引，采用tf.kears.pad_sequence将句子长度进行统一
def process_file(filename,word_to_id,cat_to_id,max_length = 600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])#将每句话id化
        label_id.append(cat_to_id[labels[i]])#每句话对应的类别的id
    #
    # # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = krs.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = krs.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    #
    return x_pad, y_pad
    
#按照batch_size来迭代生成训练数据
def data_load(x,y,batch_size = 32):
    len_data = len(x)
    nums_batch = int((len_data-1)/batch_size) + 1
    #产生随机排序数列
    indices = np.random.permutation(np.arange(len_data))
    #将x，y随机打乱
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    
    for i in range(nums_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size,len_data)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]   