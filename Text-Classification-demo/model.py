import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

"""RNN—model"""
class TextRnn(nn.Module):
    def __init__(self):
        super(TextRnn,self).__init__()
        self.embedding = nn.Embedding(5000,64)
        self.rnn = nn.LSTM(input_size = 64,hidden_size = 128,num_layers=2,bidirectional = True)
        self.f1 = nn.Sequential(nn.Linear(256,128),
                                nn.Dropout(0.2),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128,10),
                                nn.Softmax())
    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = f.dropout(x,p = 0.2)
        x = self.f1(x[:,-1,:])
        return self.f2(x)

"""CNN-model"""
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
        x = self.embedding(x) # batch_size x text_len x embedding_size 64*600*64
        x = x.permute(0, 2, 1) #64*64*600
        x = self.conv(x)  #Conv1后64*256*596,ReLU后不变,NaxPool1d后64*256*1
        x = x.view(-1, x.size(1)) #64*256
        x = F.dropout(x, 0.8)
        x = self.f1(x)    #64*10 batch_size * class_num
        return x
