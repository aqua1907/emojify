import pandas as pd
import numpy as np
import emoji


def read_csv(file_path):
    """
    :param file_path: take file path of csv file
    :return: an array with all phrases;
             an array of labels
    """
    text = []
    label = []
    data = pd.read_csv(file_path, header=None)
    for ix, row in data.iterrows():
        text.append(row[0])
        label.append(row[1])

    return np.array(text), np.array(label)


def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    :param label: which is integer
    :return: printed emoji corresponding to the label
    """
    emoji_dictionary = {0: "\u2764\ufe0f", # red :heart:
                        1: ":baseball:",
                        2: ":smile:",
                        3: ":disappointed:",
                        4: ":fork_and_knife:"}

    return emoji.emojize(emoji_dictionary[label], use_aliases=True)


def read_glove_vecs(glove_path):
    """
    Read 50-dimensional vector of each word of 400k words
    :param glove_path: take path
    :return: index_to_words — where mapped index to the word;
             words_to_index — where mapped word to the index;
             word_to_vec_map — where mapped word to 50-d vector;
    """
    with open(glove_path, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            currWord = line[0]
            words.add(currWord)
            word_to_vec_map[currWord] = np.array(line[1:], dtype=np.float32)

        index_to_words = {}
        words_to_index = {}
        i = 1
        for word in sorted(words):
            index_to_words[i] = word
            words_to_index[word] = i
            i += 1

    return index_to_words, words_to_index, word_to_vec_map


def sentences_to_indices(X, words_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    :param X: array of sentences (strings), of shape (m, 1)
    :param words_to_index: a dictionary containing the each word mapped to its index
    :param max_len: maximum number of words in a sentence. You can assume every sentence in X is no longer than this.
    :return: array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]

    X_indices = np.zeros((m, max_len))

    for i in range(m):
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [x.lower() for x in X[i].split()]

        j = 0

        for w in sentence_words:
            X_indices[i, j] = words_to_index[w]
            j += 1

    return X_indices
