import pandas as pd
import os
import warnings
import pathlib
import re
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

import jieba
import gensim

from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models import word2vec

root = pathlib.Path(os.path.abspath('__file__')).parent

train_path = os.path.join(root, 'data', 'train.csv')
test_path = os.path.join(root, 'data', 'test.csv')

rain_data = pd.read_csv(train_path,encoding = 'utf-8')
test_data = pd.read_csv(test_path,encoding = 'utf-8')
#train_data.head()

#对文本内容进行清洗
def clean_sentence(txt):
    txt=re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|<Paragraph>|图片',
            '', txt)
    txt = re.sub(r'[0-9]{0,12}','',txt)
    txt = ''.join(txt)
    return txt

#加载停用词
def load_stop_words(stop_word_path):
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words
##加载的停用词表为哈工大的停用词表
stop_word_path = os.path.join(root,'hit_stopwords.txt')
stop_words = load_stop_words(stop_word_path)

##去除停用词
def clean_stopwords(words):
    return [word for word in words if word not in stop_words]

#对文本进行清洗后，分词处理，并去掉停用词
def sentence_process(sentence):
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = clean_stopwords(words)
    return ' '.join(words)

def data_process(data):
    for k in tqdm(['article','summarization']):
        data[k] = data[k].apply(sentence_process)
    print('Data process Done!')
    return data
start = time.time()
train_data = data_process(train_data)
test_data['article'] = test_data['article'].apply(sentence_process)
print("Word cutting is Done,take time:{}".format(time.time()-start))

merge_data = pd.concat([train_data[['article']],train_data[['summarization']],test_data],axis = 0)
merge_data.to_csv('merge_data.csv',index = None,header = False)
train_data.to_csv('train_seg_data.csv')
test_data.to_csv('test_seg_data.csv')
print('train_data size {},test_data size {}，merge_data size {}'.format(len(train_data),len(test_data),len(merge_data)))

merge_data.drop(['summarization'],axis=1,inplace=True)
merge_data.to_csv('merge_data.csv',index = None,header = False)