import collections
import numpy as np
import math
import os
import random
import time

from utils import *

class Dataset(object):
    END = ['eoood']
    def __init__(self, data_files, vocab_size_limit, partition=[.8, .1, .1], sentence_labels=(-1,1), save_path = ''):
        if sum(partition) != 1:
            error("partitions sum not correct")
        if len(data_files) != len(sentence_labels):
            error("sentence labels not properly defined: {} and {} not match".format(data_files, sentence_labels))
        self.vocab_size_limit = vocab_size_limit
        self.save_path = save_path
        self.vocab, self.labels = self.parse_sentences(data_files, sentence_labels)
        data_idx = self.build_dataset(self.vocab, self.vocab_size_limit)
        (self.train_data, self.valid_data, self.test_data), (self.train_label, self.valid_label, self.test_label) = self.split_dataset(data_idx, self.labels, partition)
        self.save_vocab()

    def parse_sentences(self, data_files, sentence_labels):
        data = list()
        for data_file, label in zip(data_files, sentence_labels):
            raw_text = open(data_file, encoding="utf8").read()
            raw_lines = [(self.parse_tokens(sentence), label) for sentence in raw_text.split("\n")]
            nonempty = [s for s in raw_lines if len(s) > 0] # nonempty sentences are kept
            data.extend(nonempty)
        random.shuffle(data)
        unzipped_object = zip(*data)
        raw_sentences, labels = list(unzipped_object)
        return raw_sentences, labels

    def parse_tokens(self, raw_sentence):
        s = [w.replace('\x01', ' ') for w in raw_sentence.split()]
        return [w for w in s if len(w) > 0] # nonempty words are kept

    def split_dataset(self, sentences, labels, partition):
        len_all = len(sentences)
        len_train = int(partition[0] * len_all)
        len_valid = int(partition[1] * len_all)
        return (sentences[:len_train], sentences[len_train:len_train+len_valid], sentences[len_train+len_valid:]), \
               (labels[:len_train], labels[len_train:len_train+len_valid], labels[len_train+len_valid:]) # data, and labels

    def idx2sentence(self, text):
        return " ".join([self.idx2word[t] for t in text])

    def get_index(self, word):
        return self.word2idx.get(word, 0)

    def build_dataset(self, words_raw, n_words):
        words = flatten2d(words_raw)

        counter = collections.Counter(words).most_common(n_words - 1)
        word_cntdict = dict(counter)

        token_length = len(words)
        frequent_words = list(word_cntdict.keys())
        unk_cnt = token_length - sum(word_cntdict.values()) # how many '<UNK>' tokens included
        word_cntdict["<UNK>"] = unk_cnt

        self.count = list(zip(word_cntdict.keys(), word_cntdict.values()))

        #build dictionary，the higher word frequency is, the top word is
        vocab = list(word_cntdict.keys())
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))

        data = [list(map(self.get_index, line)) for line in words_raw]   
        return data

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

    def save_vocab(self):
        with open(os.path.join(self.save_path, 'vocab.txt'), 'w') as f:
            for i in range(len(self.count)):
                vocab_word = self.idx2word[i]
                f.write('%s %d\n' % (vocab_word, self.count[i][1]))


class Sampler:
    def __init__(self, dataset, window_size, neg_sample_size, batch_size):
        self.dataset = dataset

        self.window_size = window_size
        self.neg_sample_size = neg_sample_size
        self.span = 2 * window_size + 1
        self.batch_size = batch_size

        self.sample_table = self.init_sample_table()

    def __call__(self, mode="tests", **kwargs):
        return self.generate_batch(mode, **kwargs)

    def init_sample_table(self):
        count = [ c[1] for c in self.dataset.count]
        pow_freq = np.array(count) ** 0.75
        pow_sum = np.sum(pow_freq)
        ratio = pow_freq / pow_sum

        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []

        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def generate_batch(self, mode="tests"):
        data = self.dataset.train_data
        label = self.dataset.train_label

        pos_u = list()
        pos_v = list()
        sentence = list()
        sentence_labels = list()
        for i, (row, polarity) in enumerate(zip(data, label)):
            for col_index in range(len(row) - self.span):
                data_buffer = row[col_index : col_index + self.span]
                context = data_buffer[:self.window_size] + data_buffer[self.window_size+1:]
                target = data_buffer[self.window_size]

                pos_u.extend([target for _ in context])
                pos_v.extend(context)

            tmp_data_size = len(pos_u)

            sentence.append(np.array(row))
            sentence_labels.append(polarity)

            if (i + 1) % self.batch_size == 0 and tmp_data_size > 0:
                neg_v = np.random.choice(self.sample_table, size=(tmp_data_size, self.neg_sample_size))
                yield (np.array(pos_u), np.array(pos_v), neg_v), (sentence, sentence_labels)
                pos_u = list()
                pos_v = list()
                sentence = list()
                sentence_labels = list()





