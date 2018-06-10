import numpy as np
import os
keywords_vocab = []
positive_words = []
negative_words=[]

def get_positive_words():
    positive_words = []
    with open("data/positive-words.txt", 'r', encoding='ISO-8859-1') as f:
        for line in f:
            str = line[:-1]
            if len(str) > 0 and ";" not in str:
                positive_words.append(str)
    return positive_words


def get_negative_words():
    negative_words = []
    with open("data/negative-words.txt",'r', encoding='ISO-8859-1') as f:
        for line in f:
            str = line[:-1]
            if len(str) > 0 and ";" not in str:
                negative_words.append(str)
    return negative_words


def read_words_from_file(filename):
    input_str = ""
    with open(filename) as f:
        input_str += "".join(f.readlines()).replace("\n", " ")
    words = input_str.lower().split()
    return words

def process_words(cur_file, input_nn, all_words):
    words = read_words_from_file(cur_file)
    positive_count = 0
    negative_count = 0
    inputs = np.zeros(len(all_words))
    for word in words:
        if word in positive_words:
            positive_count += 1
            idx = all_words.index(word)
            inputs[idx] += 1
        elif word in negative_words:
            negative_count += 1
            idx = all_words.index(word)
            inputs[idx] += 0
    input_nn.append(inputs)
    return input_nn

def word_extract():
    print("script is running...")
    global positive_words
    positive_words = get_positive_words()
    global negative_words
    negative_words = get_negative_words()
    global keywords_vocab
    keywords_vocab = positive_words + negative_words
    print("=> loaded", len(positive_words), "positive words.")
    print("=> loaded", len(negative_words), "negative words.")

def load_inputs(folder, label, count, input_nn,targets):
    file_count = 0
    for cur_file in os.listdir(folder):
        if cur_file.endswith(".txt"):
            input_nn = process_words(folder + "/" + cur_file, input_nn, keywords_vocab)
            targets.append(label)
            file_count += 1
            if file_count % (count/10) == 0:
                print("=> processed", file_count, folder, "files.")

            if file_count >= count:
                break
    return input_nn,targets

