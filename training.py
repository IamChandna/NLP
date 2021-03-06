from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
top_words = 6790
max_words = 6790

def get_model(X, y):
    mod = Sequential()
    # mod.add(Embedding(top_words, 32, input_length=max_words))
    mod.add(Dense(1, input_dim=X.shape[1]))
    mod.add(Activation('sigmoid'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    mod.compile(loss='mean_squared_error', optimizer=sgd)
    mod.fit(X, y, nb_epoch=200, batch_size=1)
    return mod