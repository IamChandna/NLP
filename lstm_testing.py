from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.datasets import imdb
from random import shuffle

# from model3 import X_test, y_test


# Using keras to load the dataset with the top_words
top_words = 80000
max_review_length = 250

# (X_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
# x_test = [[11,1806,1341,184,1473,343,19,67,4219,790,1,830,4,1,509]]
# x_test = [[11,13,21,49,1,106,27,141,456,41,6,3,52,9423,32660,13059,15969]]
# x_test=[[10,89,37,11,17,17,13,75]]
# s = "This academy award winning movie can rank among the greatest of the genre "
# s = "the movie has bad content and worst actors I do not like the movie"
# s = "film was awesome having good script "
s="the movie was good and has good content with good plot"
# s = "i do not love this movie much But it has good acting skills"
s = s.lower().split()
shuffle(s)
word2id = imdb.get_word_index()
x_test = [[word2id.get(i) for i in s]]

x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# load json and create model arch
json_file = open('lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('lstm.h5')

# scores = model.evaluate(X_test, y_test, verbose=0)
scores = model.predict(x_test)>0.5
print(scores)
# def evaluate_model():
