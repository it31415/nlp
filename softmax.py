import numpy as np
from keras.layers import Input
from keras.layers.core import Activation
from keras.models import Model


x = Input(shape=(20, 5))
# xにsoftmaxを作用させてください
y = Activation('softmax')(x)

model = Model(inputs=x, outputs=y)

sample_input = np.ones((12, 20, 5))
sample_output = model.predict(sample_input)

print(np.sum(sample_output, axis=2))
