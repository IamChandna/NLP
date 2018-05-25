import utils
import training
import numpy as np
from random import shuffle

if __name__ == '__main__':
    utils.word_extract()

    input_nn = []
    targets = []

    samples_per_category = 5000
    a,b = utils.load_inputs("aclImdb/train/neg", 0, samples_per_category, input_nn, targets)
    input_nn=list(a)
    targets=list(b)
    a,b = utils.load_inputs("aclImdb/train/pos", 1, samples_per_category, input_nn, targets)
    input_nn.extend(list(a))
    targets.extend(list(b))
    input_nn_shuffle = []
    targets_shuffle = []
    index_shuffle = list(range(len(input_nn)))
    shuffle(index_shuffle)
    for i in index_shuffle:
        input_nn_shuffle.append(input_nn[i])
        targets_shuffle.append(targets[i])

    num_categories = 2
    cutoff = int(float(samples_per_category * num_categories) * 2 / 3)
    input_nn_shuffle = np.array(input_nn_shuffle)
    targets_shuffle = np.array(targets_shuffle)

    input_nn_train = input_nn_shuffle[:cutoff]
    targets_train = targets_shuffle[:cutoff]

    input_nn_valid = input_nn_shuffle[cutoff:]
    targets_valid = targets_shuffle[cutoff:]

    model = training.get_model(input_nn_train, targets_train)

    predict_train = model.predict(input_nn_train) > 0.5
    print("training=", np.mean(np.array(predict_train[:, 0], dtype=int) == targets_train))

    predict_valid = model.predict(input_nn_valid) > 0.5
    print("validation=", np.mean(np.array(predict_valid[:, 0], dtype=int) == targets_valid))
