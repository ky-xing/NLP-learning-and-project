import pandas as pd
import numpy as np
import torch
from torch import nn
from data_loader import read_category,read_vocab,process_file,data_load
import tensorflow as tf
import tensorflow.keras as krs
from model import TextRnn

contents = 'cnews_test.txt'
class PredictModel:
    def __init__(self):
        self.categories,self.cat_to_id = read_category()
        self.words,self.word_to_id = read_vocab('cnews_vocab.txt')
        ##直接加载训练过的模型
        ##self.model = torch.load('\model.pkl')
        ##也可以选择加载模型之后，加载参数
        self.model = TextRnn()
        self.model.load_state_dict(torch.load('model_params.pkl'))
    def predict(self,message):
        content = str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        data = krs.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred = self.model(data)
        class_index = torch.argmax(y_pred[0]).item()
        return self.categories[class_index]

if __name__ ==  '__main__':
    model = PredictModel()
    test_news = ['专家预测最后9战6胜3负 火箭闯入前八概率只三成新浪体育讯北京时间3月29日消息，美国网站ESPN专家约翰-霍林格给出了他自己的季后赛出线预测，根据他的预测，主要竞争西部季后赛席位的几支球队晋级概率如下：开拓者96.3%、黄蜂93.0%、灰熊87.5%、火箭22.6%、太阳0.6%。',
    '地铁沿线将集中供地晨报讯(记者 赵阳)地价和房价的密切关系，使明年本市供地状况成为关注热点。']
     
    for i in test_news:
        calss_news = model.predict(i)
        print('The Class of news :',calss_news)
     