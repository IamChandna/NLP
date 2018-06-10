from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# Embedding
max_features = 80000
maxlen = 250
embedding_size = 300

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 32
epochs = 3

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')



print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
# model.add(LSTM(lstm_output_size))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
hist=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

train_val_accuracy = hist.history
train_acc = train_val_accuracy['acc']
val_acc = train_val_accuracy['val_acc']
print('          Done!')
print('     Train acc: ', train_acc[-1])
print('Validation acc: ', val_acc[-1])
print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])


# saving model
model_json = model.to_json()
with open("lstm" + ".json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
model.save_weights("lstm" + ".h5")
print("Saved model to disk")

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)