from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from data_process import load_text_data
from metrics import micro_f1,macro_f1
from sklearn.metrics import classification_report

def test(model,x_test,y_test):

    print('Start Testing......')
    y_pred = model.predict(x_test)

    y_pred = tf.constant(y_pred,tf.float32)
    y_test = tf.constant(y_test,tf.float32)

    print(micro_f1(y_test,y_pred))
    print(macro_f1(y_test,y_pred))

    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    x_train,x_valid,x_test,y_train,y_valid,y_test = load_text_data(95)
    #model = load_model('./data/TextCNN_model.h5')
    model = load_model('./data/Transformer_model.h5')
    test(model,x_test,y_test)