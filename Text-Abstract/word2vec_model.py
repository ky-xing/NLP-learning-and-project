import pandas as pd
import os
import warnings
import pathlib
import re
warnings.filterwarnings('ignore')

from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import word2vec
#获取根目录
root = pathlib.Path(os.path.abspath('__file__')).parent

train_path = os.path.join(root, 'train_seg_data.csv')
test_path = os.path.join(root, 'test_seg_data.csv')
#读取分词后的数据
train_data = pd.read_csv(train_path,encoding = 'utf-8')
test_data = pd.read_csv(test_path,encoding = 'utf-8')

merge_data  = pd.read_csv('merge_data.csv',header = None)

#训练词向量

##词向量维度为128维
#sg=1采用skip-gram的方法训练
#negetive=5每次取5个词作为负样本
#窗口长度为5
#取词的最小词频5，去除一些不常用的词
word2vec_model = Word2Vec(LineSentence('merge_data.csv'),sg = 1,size = 300 ,negative=5,workers = 8,window =5,min_count=5)

##提取word2vec获得的词表
vocab = word2vec_model.wv.vocab

##获取最大长度
def get_max_len(data):
    # TODO FIX len size bug
    max_lens = data.apply(lambda x: x.count(' ') + 1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))

## 填充字段，句首添加'start'，句末添加'end'，长度不足的添加'pad'，未出现在词表中的词填充'UNK'
def pad_process(sentence,max_len,vocab):
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence) 

#将文本数据的X进行处理
train_x_max_len = get_max_len(train_data['article'])
test_x_max_len = get_max_len(test_data['article'])
X_max_len = max(train_x_max_len,test_x_max_len)
train_data['X'] = train_data['article'].apply(lambda x: pad_process(x, X_max_len, vocab))
test_data['X'] = test_data['article'].apply(lambda x: pad_process(x, X_max_len, vocab))
#将文本数据的Y进行处理
train_y_max_len = get_max_len(train_data['summarization'])
train_data['Y'] = train_data['summarization'].apply(lambda x: pad_process(x, train_y_max_len, vocab))
#保存处理后的数据
train_data['X'].to_csv('train_x_pad.csv', index=None, header=False)
train_data['Y'].to_csv('train_y_pad.csv', index=None, header=False)
test_data['X'].to_csv('test_x_pad.csv', index=None, header=False)

#词向量再次训练
print('start retrain w2v model')
word2vec_model.build_vocab(LineSentence('train_x_pad.csv'), update=True)
word2vec_model.train(LineSentence('train_x_pad.csv'), epochs=5, total_examples=word2vec_model.corpus_count)
print('1/3 Done！')
word2vec_model.build_vocab(LineSentence('train_y_pad.csv'), update=True)
word2vec_model.train(LineSentence('train_y_pad.csv'), epochs=5, total_examples=word2vec_model.corpus_count)
print('2/3 Done！')
word2vec_model.build_vocab(LineSentence('test_x_pad.csv'), update=True)
word2vec_model.train(LineSentence('test_x_pad.csv'), epochs=5, total_examples=word2vec_model.corpus_count)
print('3/3 Done！')


vocab = {word: index for index, word in enumerate(word2vec_model.wv.index2word)}
reverse_vocab = {index: word for index, word in enumerate(word2vec_model.wv.index2word)}

word2vec_model.save('word2vec.model')
print('finish retrain w2v model')
print('final w2v_model has vocabulary of ', len(word2vec_model.wv.vocab))

#保存词表
def save_vocab(file_path, data):
    with open(file_path) as f:
        for i in data:
            f.write(i)
#保存词典和对应的索引
def save_dict(save_path, dict_data):
    """
    保存字典
    :param save_path: 保存路径
    :param dict_data: 字典路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for k, v in dict_data.items():
            f.write("{}\t{}\n".format(k, v))

vocab_path = os.path.join(root,'vocab.txt')
reverse_vocab_path = os.path.join(root, 'reverse_vocab.txt')
save_dict(vocab_path, vocab)
save_dict(reverse_vocab_path, reverse_vocab)

#保存训练后的词向量矩阵
embedding_matrix = word2vec_model.wv.vectors
embedding_matrix_path = os.path.join(root,'embedding_matrix.txt')
np.savetxt(embedding_matrix_path, embedding_matrix, fmt='%0.8f')

def transform_data(sentence, vocab):
    # 字符串切分成词
    words = sentence.split(' ')
    # 按照vocab的index进行转换         # 遇到未知词就填充unk的索引
    ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return ids

# 将词转换成索引  [<START> 广东省 人大 常委会 ...] -> [32800, 403, 986, 246, 231]
train_id_x = train_data['X'].apply(lambda x: transform_data(x, vocab))
train_id_y = train_data['Y'].apply(lambda x: transform_data(x, vocab))
test_id_x = test_data['X'].apply(lambda x: transform_data(x, vocab))

# 将索引列表转换成矩阵
train_X = np.array(train_id_x.tolist())
train_Y = np.array(train_id_y.tolist())
test_X = np.array(test_id_x.tolist())
# 保存数据
np.savetxt('train_x.txt', train_X, fmt='%0.8f')
np.savetxt('train_y.txt', train_Y, fmt='%0.8f')
np.savetxt('test_x.txt', test_X, fmt='%0.8f')