import numpy as np
from keras.layers import Input, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import dot, concatenate
from keras.layers.core import Activation
from keras.models import Model

batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length1 = 20 # 文1の長さ
seq_length2 = 30 # 文2の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数
hidden_dim = 200 # 最終出力のベクトルの次元数

# 2つのLSTMに共通のEmbeddingLayerを使うため、はじめにEmbeddingLayerを定義します。
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

input1 = Input(shape=(seq_length1,))
embed1 = embedding(input1)
bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed1)

input2 = Input(shape=(seq_length2,))
embed2 = embedding(input2)
bilstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed2)

# 要素ごとの積を計算する
product = dot([bilstm2, bilstm1], axes=2) # サイズ：[バッチサイズ、文の長さ、文の長さ]

# ここにAttention mechanismを実装してください
a = Activation('softmax')(product)
c = dot([a, bilstm1], axes=[2, 1])
c_bilstm2 =  concatenate([c, bilstm2], axis=2)
h = Dense(hidden_dim, activation='tanh')(c_bilstm2)

model = Model(inputs=[input1, input2], outputs=h)

sample_input1 = np.arange(batch_size * seq_length1).reshape(batch_size, seq_length1)
sample_input2 = np.arange(batch_size * seq_length2).reshape(batch_size, seq_length2)

sample_output = model.predict([sample_input1, sample_input2])
print(sample_output.shape)
