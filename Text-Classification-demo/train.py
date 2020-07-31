import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
from data_loader import read_category,read_vocab,process_file,data_load
from model import TextRnn
from time import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

##查看GPU是否可用
print(torch.cuda.is_available())

categories, cat_to_id = read_category()
print(categories)

words, word_to_id = read_vocab('cnews_vocab.txt')
print(words)
##加载训练集 
x_train, y_train = process_file('cnews_small_sample.txt', word_to_id, cat_to_id, 600)
print('x_train=', x_train)
##加载验证集
x_val, y_val = process_file('cnews_val.txt', word_to_id, cat_to_id, 600)


###验证集上进行准确率评估
def evaluate(model, Loss, optimizer, x_val, y_val):
    
    batch_val = data_load(x_val, y_val, 32)
    acc = 0
    los = 0
    for x_batch, y_batch in batch_val:
        size = len(x_batch)
        x = np.array(x_batch)
        y = np.array(y_batch)
        x = torch.LongTensor(x)
        y = torch.Tensor(y)
        x = Variable(x).cuda()#采用GPU训练，将x放入cuda中
        y = Variable(y).cuda()#采用GPU训练，将y放入cuda中
        out = model(x)
        loss = Loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = np.mean(loss.data.cpu().numpy())
        accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
        acc +=accuracy*size
        los +=loss_value*size
    return los/len(x_val), acc/len(x_val)

model = TextRnn()
#model = TextCNN()
model.cuda()##采用GPU训练
Loss = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
best_val_acc = 0
print('Start Training:')
for epoch in range(100):
    i = 0
    print('epoch:{}'.format(epoch))
    start_time = time()
    batch_train = data_load(x_train,y_train,32)
    for x_batch,y_batch in tqdm(batch_train):
        i +=1
        x = np.array(x_batch)
        y = np.array(y_batch)
        x = torch.LongTensor(x).cuda()
        y = torch.Tensor(y).cuda()
        out = model(x)
        loss = Loss(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if i %90 ==0:
            los,accuracy = evaluate(model,Loss,optimizer,x_val,y_val)
            print('loss:{},accuracy{}'.format(los,accuracy))
            if accuracy > best_val_acc:
                torch.save(model.state_dict(),'model_params.pkl')
                best_val_acc = accuracy
    end_time = time()
    print('time:',end_time-start_time)
print('Training Finished!')
#保存模型
torch.save(model,'\model.pkl')