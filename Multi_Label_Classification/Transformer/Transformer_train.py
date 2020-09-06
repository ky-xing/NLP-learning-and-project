import os

import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Transformer_model import Transformer
from metrics import micro_f1,macro_f1
from utils import create_padding_mask
from layers import CustomSchedule
from sklearn.metrics import f1_score
from tqdm import tqdm

from collections import namedtuple,defaultdict
params=namedtuple('params',['data_path','vocab_save_dir','vocab_size','padding_size'])
params.data_path='./data/cleaning_data_95.csv'
params.vocab_save_dir='./data/vocab.txt'
params.vocab_size=29299  ##词表长度
params.padding_size=150  ##输入最大序列长度
params.BUFFER_SIZE=3000  ##维持shuffle最大元素个数
params.BATCH_SIZE=128    

from data_process import load_text_data
from sklearn.preprocessing import MultiLabelBinarizer
### 加载数据,将数据放入Dataset
def load_dataset():
    x_train,x_valid,x_test,y_train,y_valid,y_test = load_text_data(95)

    ###使用Dataset生成迭代数据
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    # 将数据集缓存到内存中以加快读取速度。
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(params.BUFFER_SIZE,reshuffle_each_iteration=True).batch(params.BATCH_SIZE, 
                                        drop_remainder=True)

    valid_dataset=valid_dataset.batch(params.BATCH_SIZE)
    # 流水线技术 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，CPU 开始准备步骤 N + 1 的数据。
    # 这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。
    # 没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态:
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 测试一下
    x, y = next(iter(train_dataset))
    print(x.shape)
    print(y.shape)
    return train_dataset,valid_dataset


###使用 input_signature 指定通用形状
train_step_signature = [
    tf.TensorSpec(shape=(None, 150), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]
# 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  
    enc_padding_mask = create_padding_mask(inp)
  
    with tf.GradientTape() as tape:
        predictions = transformer(inp,training=True,enc_padding_mask=enc_padding_mask)
        loss = loss_function(tar, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
    train_loss(loss)
    train_accuracy(tar, predictions)
    
    mi_f1=micro_f1(tar, predictions)
    ma_f1=macro_f1(tar, predictions)
    return mi_f1 ,ma_f1

def predict(inp,tar,enc_padding_mask):
    predictions = transformer(inp,False,enc_padding_mask=enc_padding_mask)
    mi_f1=micro_f1(tar, predictions)
    ma_f1=macro_f1(tar, predictions)
    return mi_f1,ma_f1

def evaluate(test_dataset):
    predictions=[]
    tars=[]
    for (batch, (inp, tar)) in tqdm(enumerate(test_dataset)):
        enc_padding_mask = create_padding_mask(inp)
        predict = transformer(inp,False,enc_padding_mask=enc_padding_mask)
        predictions.append(predict)
        tars.append(tar)
    predictions=tf.concat(predictions,axis=0)
    tars=tf.concat(tars,axis=0)
    mi_f1=micro_f1(tars, predictions)
    ma_f1=macro_f1(tars, predictions)
    
    predictions=np.where(predictions>0.5,1,0)
    tars=np.where(tars>0.5,1,0)
    
    smaple_f1=f1_score(tars,predictions,average='samples')
    return mi_f1,ma_f1,smaple_f1,tars,predictions

if __name__ == '__main__':
    
    train_dataset ,valid_dataset = load_dataset()
    ##定义模型超参
    num_layers = 4  #采用4层encoder
    d_model = 128   #嵌入维度
    dff = 512       #全连接层神经元数
    num_heads = 8   #self attention 个数

    input_vocab_size = 29299    #词表大小
    output_dim = 95             #输出标签数
    dropout_rate = 0.1          #dropout比例
    maximum_position_encoding=10000 #最大位置编码

    EPOCHS = 10

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                             input_vocab_size, output_dim, 
                             maximum_position_encoding, 
                             rate=dropout_rate)

    loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

    for epoch in range(EPOCHS):
        start = time.time()
    
        train_loss.reset_states()
        train_accuracy.reset_states()
    
        print('Start Train......') 
        for (batch, (inp, tar)) in enumerate(train_dataset):
            time1 = time.time()
            mic_f1,mac_f1=train_step(inp, tar)

            if batch % 50 == 0:
                test_input,test_target= next(iter(valid_dataset))
                enc_padding_mask = create_padding_mask(test_input)
                val_mic_f1,val_mac_f1=predict(test_input,test_target,enc_padding_mask)
                
                print ('Epoch {} Batch {} Loss {:.4f} micro_f1 {:.4f} macro_f1 {:.4f} val_micro_f1 {:.4f} val_macro_f1 {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), mic_f1, mac_f1,val_mic_f1,val_mac_f1))
                print('Cost time:{}'.format(time.time()-time1))
        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
        
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    transformer.save('Transformer_model.h5')
    print('Training Finished!')

    mi_f1,ma_f1,smaple_f1,tars,predictions = evaluate(valid_dataset)
    
    print(mi_f1,ma_f1,smaple_f1)