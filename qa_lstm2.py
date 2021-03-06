from keras.layers import Input, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import dot, concatenate
from keras.layers.core import Activation
from keras.models import Model

batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length1 = 20 # 質問の長さ
seq_length2 = 10 # 回答の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数
hidden_dim = 200 # 最終出力のベクトルの次元数

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

model = Model(inputs=[input1, input2], outputs=h)
model.summary()
