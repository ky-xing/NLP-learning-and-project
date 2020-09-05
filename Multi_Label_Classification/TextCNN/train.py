from TextCNN_model import TextCNN
from data_process import load_text_data
from pprint import pprint
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from metrics import micro_f1, macro_f1






def build_model():
    #定义模型参数
    feature_size=150        #padding后文本长度
    max_token_num = 29299   #词表长度
    embed_size=300          #词向量维度
    num_classes=len(y_train[0])   #标签数量
    filter_sizes=[3,4,5]    #卷积核尺寸
    dropout_rate=0.5        #dropout率控制过拟合
    regularizers_lambda=0.01#正则化率
    learning_rate=0.001     #学习率 
    batch_size=64           
    epochs=10
    embedding_matrix = np.loadtxt('./data/embedding_matrix.txt')

    model = TextCNN(max_sequence_length=feature_size, max_token_num=29299, embedding_dim=embed_size,
                    kernel_size = filter_sizes,output_dim=num_classes,embedding_matrix=embedding_matrix)
    
    model.compile(tf.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',     ##若是多分类，loss要使用categorical_crossentropy多类别交叉熵
                      metrics=[micro_f1, macro_f1])   ##micro_f1,macro_f1为自定义的F1-score
    model.summary()

    return model

def train():
    model = build_model()

    print('Start Training......')
    #早停法
    early_stopping = EarlyStopping(monitor='val_micro_f1', patience=10, mode='max')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        workers=workers,
                        use_multiprocessing=True,
                        callbacks=[early_stopping],
                        validation_data=(x_valid, y_valid))

    print("Saving model...")
    tf.keras.models.save_model(model,'./data')
    pprint(history.history)

if __name__ == '__main__':
    #加载处理过的数据
    x_train,x_valid,x_test,y_train,y_valid,y_test = load_text_data(95)

    train()