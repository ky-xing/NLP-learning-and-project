import tensorflow as tf
import os
import time
import numpy as np
from Seq2seq_model import Encoder,Decoder,Attention
import warnings
warnings.filterwarnings("ignore")

train_x = np.loadtxt('train_x.txt')
train_y = np.loadtxt('train_y.txt')
test = np.loadtxt('test_x.txt')
# 加载词向量模型
def load_word2vec_file(save_wv_model_path):
    wv_model = Word2Vec.load(save_wv_model_path)
    embedding_matrix = wv_model.wv.vectors
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    return embedding_matrix，vocab, reverse_vocab

#加载 embedding_matrix
def load_embedding_matrix():
    return np.loadtxt(embedding_matrix_path)



pad_index = vocab['<PAD>']
#定义mask损失函数，忽略掉pad部分的损失
def loss_function(real, pred):
    #统计真实值中pad的部分,pad_index = 5,real = [1,2,3,4,5] ——> [0,0,0,0,1]
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))
    loss_ = loss_object(real, pred)
    #转换mask的数据类型，与loss_的一样
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([vocab['<START>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # decoder(x, hidden, enc_output)
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # 使用teacher forcing，将前边所有targ都作为输入
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        #计算梯度
        gradients = tape.gradient(loss, variables)
        #应用处理后的梯度
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

def train_model(train_x, train_y, BATCH_SIZE):
    EPOCHS = 10
    BUFFER_SIZE = len(train_x)
    step_per_epoch = len(train_x)//BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    for epoch in range(EPOCHS):
        start_time = time.time()
        ##初始化隐藏层参数
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        ##分批训练
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            #每5个batch打印loss
            if batch % 5 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
##开始
if __name__ == '__main__':
    
    #加载embeddingmatrix，vocab
    embedding_matrix，vocab, reverse_vocab = load_word2vec_file('word2vec.model')
    # 输入的长度, 即单个样本的序列长度 
    input_length = train_x.shape[1]
    # 输出的长度  
    output_sequence_length = train_y.shape[1]
    # 词表大小
    vocab_size=len(vocab)

    ### 超参数设置 ###
    BUFFER_SIZE = len(train_x)
    BATCH_SIZE = 64
    # 词向量维度
    embedding_dim = 128
    # 隐藏层单元数
    units = 512
    # 词表大小
    vocab_size = len(vocab)

    
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_model(train_X, train_Y, BATCH_SIZE)

    
