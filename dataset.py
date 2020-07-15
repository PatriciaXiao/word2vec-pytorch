import collections
import numpy as np
import math
import os
import random
import time
from six.moves import xrange
from utils import *

class PairedDataset:
    def __init__(self, text_group1, text_group2, group_names=("democratic", "democratic"), partition=[.8, .1, .1], vocab_size_limit = 0, max_seq_length=200, sentence_labels=[-1,1]):
        # check validity
        if len(group_names) != 2: # republican
            error("wrong group names")
        if sum(partition) != 1:
            error("partitions sum not correct")

        self.group_names = group_names
        self.sentence_labels = sentence_labels
        self.max_seq_length = max_seq_length

        sentences_sets1 = self.parse_sentences(text_group1, partition) # democratic set
        sentences_sets2 = self.parse_sentences(text_group2, partition) # republican set
        # the words
        train_raw_data = [sentences_sets1[0], sentences_sets2[0]]
        valid_raw_data = [sentences_sets1[1], sentences_sets2[1]]
        tests_raw_data = [sentences_sets1[2], sentences_sets2[2]]

        # the vocabulary
        text = flatten3d(train_raw_data) # + flatten3d(self.valid_raw_data) + flatten3d(self.tests_raw_data)
        counter = collections.Counter(text)
        if vocab_size_limit <= 0:
            vocab_size_limit = len(text)
        word_cntdict = dict(counter.most_common(vocab_size_limit))
        token_length = len(text)
        frequent_words = list(word_cntdict.keys())
        self.vocab = ["<UNK>"] + frequent_words
        self.vocab_size = len(self.vocab)

        self.index_range = list(range(self.vocab_size))
        self.count = np.array([token_length - sum(word_cntdict.values())] + list(word_cntdict.values()))
        # probability of sampling the environment words
        tmp = self.count ** .75
        self.prob_sampled = tmp / sum(tmp)
        self.drop_prob_dict = self.get_drop_prob_dict(token_length)

        # the dictionary
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for i, w in enumerate(self.vocab)}

        # the indexes
        self.train_data = [[list(map(self.get_index, sentence)) for sentence in raw_data] for raw_data in train_raw_data]
        self.valid_data = [[list(map(self.get_index, sentence)) for sentence in raw_data] for raw_data in valid_raw_data]
        self.tests_data = [[list(map(self.get_index, sentence)) for sentence in raw_data] for raw_data in tests_raw_data]
        self.text_dict = {
                "train": self.train_data,
                "valid": self.valid_data,
                "tests": self.tests_data
            }

    def padding(self, sentence):
        len_sentence = len(sentence)
        if len_sentence >= self.max_seq_length:
            return sentence[:max_seq_length]
        else:
            return sentence[:] + [0] * (self.max_seq_length - len_sentence)

    def get_drop_prob_dict(self, token_length):
        drop_prob = 1 - np.sqrt(1e-5 * token_length / self.count)
        drop_prob[np.isneginf(drop_prob)] = 1.
        return dict(zip(self.index_range, drop_prob))


    def parse_sentences(self, text, partition):
        raw_sentences = [self.parse_tokens(sentence) for sentence in text.split("\n")]
        nonempty = [s for s in raw_sentences if len(s) > 0] # nonempty sentences are kept
        len_all = len(nonempty)
        len_train = int(partition[0] * len_all)
        len_valid = int(partition[1] * len_all)
        return nonempty[:len_train], nonempty[len_train:len_train+len_valid], nonempty[len_train+len_valid:]


    def parse_tokens(self, sentence):
        raw_sentence = [w.replace('\x01', ' ') for w in sentence.split()]
        return [w for w in raw_sentence if len(w) > 0] # nonempty words are kept

    def prepare_text(self, mode="tests", labels=[-1, 1]):
        '''
        prepare text for samplers
        '''
        options = ["train", "valid", "tests"]
        if mode not in options:
            error("options ({}) aren't including {}".format(options, mode))
        raw_data = self.text_dict[mode]
        labels = np.array([self.sentence_labels[0]] * len(raw_data[0]) + [self.sentence_labels[1]] * len(raw_data[1]))
        text_data = np.array(flatten2d(raw_data))
        # add some randomness
        shuffled_index = list(range(len(labels)))
        random.shuffle(shuffled_index)
        text_data = text_data[shuffled_index]
        labels = labels[shuffled_index]
        return text_data, labels

    def get_random_word(self, items=1):
        # the real word, example: "process"
        # return random.sample(self.vocab[1:], 1)[0]
        return random.choices(self.vocab[1:], weights=self.prob_sampled[1:], k=items)

    def get_random_index(self, items=1):
        # return random.randint(1, self.vocab_size)
        return random.choices(self.index_range[1:], weights=self.prob_sampled[1:], k=items)

    def get_index(self, word):
        return self.w2i.get(word, 0)

    def sentence(self, text):
        return " ".join([self.i2w[t] for t in text])

# sampler
class Dataset:
    def __init__(self, batch_size=1, window_size=2, word_pair_labels=[-1,1], drop=False):
        dataset = PairedDataset(text_group1=open("toy", encoding="utf8").read(), \
                                text_group2=open("toy", encoding="utf8").read())
        self.dataset = dataset
        self.window_size = window_size
        self.neg_sample_rate = 4.
        self.word_pair_labels = word_pair_labels
        self.batch_size = batch_size
        self.drop = drop is not False

        self.sampler = self.generate_batch

        self.idx2word = self.dataset.i2w

    def __call__(self, mode="tests", **kwargs):
        return self.sampler(mode, **kwargs)

    def generate_batch(self, mode="train"):
        sentence_text = list()
        sentence_labels = list()
        pos_u = list()
        pos_v = list()
        neg_v = list()
        current_text, current_labels = self.dataset.prepare_text(mode) # label is party label
        # for text in current_text:
        #     print(self.dataset.sentence(text))
        # exit(0)
        for batch_id, (text, polarity_label) in enumerate(zip(current_text, current_labels)):
            if len(text) < 2: continue # meaningless if not even two words in a sentence
            # add sentense text
            sentence_text.append(self.dataset.padding(text))
            sentence_labels.append(polarity_label)
            context = list()
            # print(self.dataset.sentence(text))
            # exit(0)
            if len(text) >= self.window_size * 2 + 1: # if it is long enough
                for i in range(self.window_size, len(text) - self.window_size):
                    context = text[i-self.window_size:i] + text[i+1:i+self.window_size]
                    pos_v.extend(context)
                    pos_u.extend([text[i] for _ in range(len(context))])
                    # negative sampling (sample rate can be set)
                    neg_v.extend([self.dataset.get_random_index(int(self.neg_sample_rate)) for _ in range(len(context))])
            if len(context) == 0: # otherwise
                i, j = random.sample(range(len(text)),  2)
                pos_v.append(text[i])
                pos_u.append(text[j])
                neg_v.append(self.dataset.get_random_index(int(self.neg_sample_rate)))
            # random.shuffle(raw_data)
            if (batch_id + 1) % self.batch_size == 0:
                # yield (np.array(pos_u), np.array(pos_v), np.array(neg_v)), (sentence_text, sentence_labels)
                yield np.array(pos_u), np.array(pos_v), np.array(neg_v)
                sentence_text = list()
                sentence_labels = list()
                pos_u = list()
                pos_v = list()
                neg_v = list()
