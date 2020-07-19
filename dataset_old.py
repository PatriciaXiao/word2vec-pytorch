import collections
import numpy as np
import math
import os
import random
import time
from six.moves import xrange

from utils import *

class Dataset(object):
    def __init__(self, data_file, vocab_size, window_size, neg_sample_size, batch_size):
        self.vocab_size = vocab_size
        self.save_path = ''
        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.span = 2 * window_size + 1
        self.batch_size = batch_size
        self.vocab = self.read_data(data_file)
        data_idx, self.count, self.idx2word = self.build_dataset(self.vocab, self.vocab_size)
        self.train_data = data_idx
        self.sample_table = self.init_sample_table()
        self.save_vocab()

    def read_data(self, data_file):
        data = list()
        with open(data_file) as f:
            lines = f.read().split('\n')
            for line in lines:
                data_line = [x for x in line.split(' ') if x != 'eoood']
                data.append(data_line)
        return data

    def sentence(self, text):
        return " ".join([self.idx2word[t] for t in text])

    def build_dataset(self, words_raw, n_words):
        words = flatten2d(words_raw)
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()

        #build dictionary，the higher word frequency is, the top word is
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0

        #dataset labelled
        for words in words_raw:
            tmp_index = list()
            for word in words:
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0
                    unk_count += 1
                tmp_index.append(index)
            data.append(tmp_index)


        count[0][1] = unk_count
        reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dict

    #high frequency word subsampled
    #randomly discard common words, and keep the same frequency ranking
    def subsampling(self, data):
        count = [c[1] for c in self.count]
        frequency = count / np.sum(count)
        prob = dict()
        t = 1e-3 #1e-5 # the fewer words kept in the dataset, the smaller this number

        #calculate discard probability
        for idx, x in enumerate(frequency):
            # y = (math.sqrt(x / 0.001) + 1) * 0.001 / x
            y = math.sqrt(t / x)
            prob[idx] = y
        subsampled_data = list()
        for line in data:
            subsampled_line = list()
            for word in line:
                if random.random() < prob[word]:
                    subsampled_line.append(word)
            subsampled_data.append(subsampled_line)
        return subsampled_data

    def init_sample_table(self):
        count = [ c[1] for c in self.count]
        pow_freq = np.array(count) ** 0.75
        pow_sum = np.sum(pow_freq)
        ratio = pow_freq / pow_sum

        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []

        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def save_vocab(self):
        with open(os.path.join(self.save_path, 'vocab.txt'), 'w') as f:
            for i in xrange(len(self.count)):
                vocab_word = self.idx2word[i]
                f.write('%s %d\n' % (vocab_word, self.count[i][1]))

    def generate_batch(self):
        data = self.train_data

        pos_u = list()
        pos_v = list()
        for row_index in range(len(data)):
            for col_index in range(len(data[row_index]) - self.span):
                data_buffer = data[row_index][col_index : col_index + self.span]
                context = data_buffer[:self.window_size] + data_buffer[self.window_size+1:]
                target = data_buffer[self.window_size]

                pos_u.extend([target for _ in context])
                pos_v.extend(context)

            tmp_data_size = len(pos_u)
            if (row_index + 1) % self.batch_size == 0 and tmp_data_size > 0:
                neg_v = np.random.choice(self.sample_table, size=(tmp_data_size, self.neg_sample_size))
                yield np.array(pos_u), np.array(pos_v), neg_v
                pos_u = list()
                pos_v = list()



