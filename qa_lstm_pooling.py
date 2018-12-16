from keras.layers import Input, Dense, Dropout, Lambda, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import dot, concatenate, subtract, multiply
from keras.layers.core import Activation
from keras.layers.pooling import AveragePooling1D
from keras import backend as K
from keras.models import Model

batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length1 = 20 # 質問の長さ
seq_length2 = 10 # 回答の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数
hidden_dim = lstm_units * 2 # 最終出力のベクトルの次元数

def abs_sub(x):
    return K.abs(x[0] - x[1])

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

input1 = Input(shape=(seq_length1,))
embed1 = embedding(input1)
bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed1)
h1 = Dropout(0.2)(bilstm1)

input2 = Input(shape=(seq_length2,))
embed2 = embedding(input2)
bilstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed2)
h2 = Dropout(0.2)(bilstm2)

# 要素ごとの積を計算する
product = dot([h2, h1], axes=2) # サイズ：[バッチサイズ、回答の長さ、質問の長さ]
a = Activation('softmax')(product)
c = dot([a, h1], axes=[2, 1])
c_h2 = concatenate([c, h2], axis=2)
h = Dense(hidden_dim, activation='tanh')(c_h2)

mean_pooled_1 = AveragePooling1D(pool_size=seq_length1, strides=1, padding='valid')(h1)
mean_pooled_2 = AveragePooling1D(pool_size=seq_length2, strides=1, padding='valid')(h)

mean_pooled_1 = Reshape((lstm_units * 2,))(mean_pooled_1)
mean_pooled_2 = Reshape((lstm_units * 2,))(mean_pooled_2)

sub = Lambda(abs_sub)([mean_pooled_1, mean_pooled_2])
mult = multiply([mean_pooled_1, mean_pooled_2])
con = concatenate([mean_pooled_1, mean_pooled_2, sub, mult], axis=-1)
con = Reshape((lstm_units * 2 * 4,))(con)
output = Dense(2, activation='softmax')(con)

model = Model(inputs=[input1, input2], outputs=output)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy")
