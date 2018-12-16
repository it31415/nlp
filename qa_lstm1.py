from keras.layers import Input, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model


vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length1 = 20 # 質問の長さ
seq_length2 = 10 # 回答の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数

embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

input1 = Input(shape=(seq_length1,))
embed1 = embedding(input1)
bilstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed1)
h1 = Dropout(0.2)(bilstm1)
model1 = Model(inputs=input1, outputs=h1)


input2 = Input(shape=(seq_length2,))
embed2 = embedding(input2)
bilstm2 = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed2)
h2 = Dropout(0.2)(bilstm2)
model2 = Model(inputs=input2, outputs=h2)

model1.summary()
model2.summary()
