import numpy as np
from keras.preprocessing.sequence import pad_sequences


# 引数にはこれを使ってください。
maxlen = 10
dtype = np.int32
padding = 'post'
truncating = 'post'
value = 0

# データ
s = [[1,2,3,4,5,6], [7,8,9,10,11,12,13,14,15,16,17,18], [19,20,21,22,23]]

# padding, truncatingをしてください。
s = pad_sequences(s, maxlen=maxlen, dtype=dtype, padding=padding, truncating=truncating, value=value)


print(s)
