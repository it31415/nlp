import numpy as np
from keras.layers import Input, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model

batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length = 20 # 文1の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数

input = Input(shape=(seq_length,))
embed = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length)(input)
bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True), merge_mode='concat')(embed)
output = Dropout(0.3)(bilstm)



model = Model(inputs=input, outputs=output)

sample_input = np.arange(batch_size * seq_length).reshape(batch_size, seq_length)
sample_output = model.predict(sample_input)

print(sample_output.shape)
