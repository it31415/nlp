import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


batch_size = 32 # バッチサイズ
vocab_size = 1000 # 扱う語彙の数
embedding_dim = 100 # 単語ベクトルの次元
seq_length = 20 # 文の長さ
lstm_units = 200 # LSTMの隠れ状態ベクトルの次元数

# 今回も簡単に入力データを準備しました。
input_data = np.arange(batch_size * seq_length).reshape(batch_size, seq_length)

# modelにBiLSTMを追加してください。
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))



# input_dataのshapeがどのように変わるのか確認してください。
output = model.predict(input_data)
print(output.shape)
