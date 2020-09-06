import tensorflow as tf

from layers import Encoder
from metrics import micro_f1,macro_f1
from utils import create_padding_mask
from layers import CustomSchedule

##只采用了encoder部分
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               output_dim, maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()
        #encoder层
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, maximum_position_encoding, rate)
        #降维
        self.x_flatten = tf.keras.layers.Flatten()     
        #全连接层  
        self.final_layer = tf.keras.layers.Dense(output_dim,activation='sigmoid')
    
    def call(self, inp, training, enc_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        flatten_output=self.x_flatten(enc_output)
        #全连接层获得输出
        final_output = self.final_layer(flatten_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output