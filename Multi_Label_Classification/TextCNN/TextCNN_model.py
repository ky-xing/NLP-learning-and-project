import logging
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def TextCNN(max_sequence_length, max_token_num, embedding_dim,kernel_size, output_dim,filter=2, model_img_path=None, embedding_matrix=None):
    """
    TextCNN:
    1.mbedding layers,可以采用预训练词向量
    2.convolution layer,采用多个不同大小的卷积核
    3.pooling layer,采用max-pooling
    4.softmax layer.多分类使用softmax激活函数，多标签分类采用sigmod激活函数
    """

    x_input = Input(shape=(max_sequence_length,))
    logging.info("x_input.shape: %s" % str(x_input.shape))  # (?, 150)

    if embedding_matrix is None:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=True)(x_input) ##trainable=Trun,词向量随训练更新 

    logging.info("x_emb.shape: %s" % str(x_emb.shape))  # (?, 150, 300)

    pool_output = []
    kernel_sizes = kernel_size ##可选kernel_size[2,3,4]或其他
    for kernel_size in kernel_sizes:
        #卷积
        c = Conv1D(filters=filter, kernel_size=kernel_size, strides=1)(x_emb)
        #maxpooling
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))
    #将maxpooling输出拼接
    pool_output = concatenate([p for p in pool_output])
    logging.info("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)

    x_flatten = Flatten()(pool_output)  # (?, 6)
    y = Dense(output_dim, activation='sigmoid')(x_flatten)  # (?, num_classes)

    logging.info("y.shape: %s \n" % str(y.shape))

    model = Model([x_input], outputs=[y])

    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    model.summary()

    return model
