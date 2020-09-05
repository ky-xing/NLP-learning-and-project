import pandas as pd
import numpy as np
import time
import warnings
import re
import jieba
warnings.filterwarnings('ignore')

def data_process(data,task):
    
    
    data['item']=data['item'].apply(sentence_process)

    print(data.head())
    print('Data cleaning Done!')

    data['item'].to_csv('sentence.csv',header = None,index=None)

    from gensim.models.word2vec import Word2Vec,LineSentence
    from gensim.models.fasttext import FastText
    """
    采用word2vec或Fasttext训练词向量
    embedding_dim = 300
    sg=1,采用skip-gram方法
    negative=5,负采样5个
    """
    wv_model = Word2Vec(LineSentence('sentence.csv'),size=300, negative=5,sg=1, workers=8, iter=20, window=5,min_count=2)
    ##wv_model = FastText(LineSentence('sentence.csv'),size=300, window=5, min_count=5, workers=4,sg=1)
    ## fasttext比较占用空间，训练时间较长
    #获取词表
    vocab = wv_model.wv.vocab
    ##文本长度取150
    x_max_len = 150


    data['item'] = data['item'].apply(lambda x: pad_sentence(x, x_max_len, vocab))
    data['item'].to_csv('sentence_1.csv',header = None,index=None)

    #对加了UNK，PAD之后的句子再次进行训练
    wv_model.build_vocab(LineSentence('sentence_1.csv'), update=True)
    wv_model.train(LineSentence('sentence_1.csv'), epochs=10, total_examples=wv_model.corpus_count)

    wv_model.save('word2vec.model')
    print('word2vec_model Saved!')
    ##获取词表+索引
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}

    

    train_x = data['item'].apply(lambda x: transform_data(x, vocab))
    x = np.array(train_x.tolist())
    #获取词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    #将词向量矩阵保存
    np.savetxt('embedding_matrix.txt', embedding_matrix, fmt='%0.8f')
    print('embedding_matrix Saved!')

    #对标签进行转换
    ### 根据不同层级标签，可将标签转换为4分类，17分类，95分类

    ##4分类，17分类为单标签多分类
    data['subject'] = data['labels'].apply(lambda x:x.split()[1])  #4分类标签
    data['topic'] = data['labels'].apply(lambda x:x.split()[2])  #17分类标签
    data.to_csv('./data/cleaning_data_95.csv',index=None)
    print('Cleaning Data Saved!')
    ## 95分类为多标签分类，转换为95个二分类
    from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
    lb = LabelEncoder()
    mlb = MultiLabelBinarizer()
    if task == '4':
        y = lb.fit_transform(data['subject'])
    elif task == '17':
        y = lb.fit_transform(data['topic'])
    elif task == '95':
        y = mlb.fit_transform(data.labels.apply(lambda x: x.split()))
    else: 
        print('Error:please input task 4,17 or 95!')
    print(len(y[0]))

    np.save('./data/x_{}.npy'.format(int(task)), x)
    np.save('./data/y_{}.npy'.format(int(task)), y)

    return x,y


def transform_data(sentence, vocab):
        """
        根据词表将文本转换为索引
        """
        # 字符串切分成词
        words = sentence.split(' ')
        # 按照vocab的index进行转换         # 遇到未知词就填充unk的索引
        ids = [vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
        return ids
##采用正则过滤无用字符，并分词
def clean_sentence(text):
    text = re.sub(
            r"[a-zA-Z0-9]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+|题目", '',text)
    words = jieba.cut(text, cut_all=False)
    return words

##加载停用词
def load_stopwords(stop_word_path):
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后空格、换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

stop_words = load_stopwords('./data/hit_stopwords.txt')

def sentence_process(sentence):
    #清洗
    words = clean_sentence(sentence)
    #过滤停用词
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def pad_sentence(sentence, max_len, vocab):
        '''
        # 填充字段 <pad> <unk> max_lens
        '''
        # 按空格统计切分出词
        words = sentence.strip().split(' ')
        # 截取规定长度的词数
        words = words[:max_len]
        #填充< unk > ,判断是否在vocab中, 不在填充 < unk >
        sentence = [word if word in vocab else '<UNK>' for word in words]
        #不足150的填充<PAD>
        sentence = sentence + ['<PAD>'] * (max_len - len(words))
        return ' '.join(sentence)

from sklearn.model_selection import train_test_split

def load_text_data(task):
    x = np.load('./data/x_{}.npy'.format(task)).astype(np.float32)
    y = np.load('./data/y_{}.npy'.format(task)).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2020)

    return x_train,x_valid,x_test,y_train,y_valid,y_test

if __name__ == '__main__':
    root = './data/data_95.csv'
    data = pd.read_csv(root)

    x,y = data_process(data,'95')
