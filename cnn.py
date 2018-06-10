from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard


# Using keras to load the dataset with the top_words
top_words = 80000
max_review_length = 250

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Pad the sequence to the same length
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)

print("Compiling the model")
#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
hist=model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=5,callbacks=[tensorBoardCallback], batch_size=64)

train_val_accuracy = hist.history
train_acc = train_val_accuracy['acc']
val_acc = train_val_accuracy['val_acc']
print('          Done!')
print('     Train acc: ', train_acc[-1])
print('Validation acc: ', val_acc[-1])
print(' Overfit ratio: ', val_acc[-1] / train_acc[-1])

# saving model
model_json = model.to_json()
with open("model" + ".json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
model.save_weights("model" + ".h5")
print("Saved model to disk")

# evaluate_model(X_test, y_test)
# Evaluation on the test set
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))