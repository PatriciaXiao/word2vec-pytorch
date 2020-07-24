import time

import torch
from torch.autograd import Variable
import torch.optim as optim

from model import SkipGram, CBOW

#DEBUG = True #False
DEBUG = False

if DEBUG:
    from dataset import Dataset
else:
    from dataset_old import Dataset, Sampler

import random
import numpy as np

import matplotlib.pyplot as plt

class word2vec:
    def __init__(self, input_file, model_name, vocabulary_size=100000,
                 embedding_dim=200, epoch=10, batch_size=256, windows_size=2, neg_sample_size=10):
        self.model_name = model_name
        if DEBUG:
            self.data_loader = Dataset(batch_size=batch_size, window_size=windows_size)
        else:
            self.data = Dataset([input_file, input_file], vocabulary_size)
            self.data_loader = Sampler(self.data, window_size=windows_size, neg_sample_size=neg_sample_size, batch_size=batch_size)
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.neg_sample_size = neg_sample_size

    def showPlot(self, points, title):
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(points)
        plt.show()

    def train(self, report=True):
        model = SkipGram(self.vocabulary_size, self.embedding_dim)

        loss_list = list()

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)

        for epoch in range(self.epoch):

            start = time.time()
            self.data.process =True
            batch_num = 0
            batch_new = 0

            for data_word, data_sentence in self.data_loader():

                optimizer.zero_grad()
                loss = model(data_word) / self.batch_size
                # loss = model(pos_u, pos_v, neg_v, self.batch_size, target, contex, labels)
                loss_list.append(loss)
                loss.backward()
                optimizer.step()

                if report and batch_num % 7 == 0: # 3000
                    end = time.time()
                    print('epoch,batch = %2d %5d:   batch_size = %5d  loss = %4.3f\r'
                          % (epoch, batch_num, self.batch_size, loss.item()), end="\n")
                    batch_new = batch_num
                    start = time.time()
                batch_num += 1

        self.showPlot(loss_list, 'Losses')
        model.save_embeddings(self.data.idx2word, 'word_embdding.txt')


if __name__ == '__main__':

    fixed_seed = 1
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.cuda.manual_seed(fixed_seed)

    # w2v = word2vec('text8', 'SkipGram')
    w2v = word2vec('toy', 'SkipGram', batch_size=1, epoch=20)
    # w2v = word2vec('democratic_cleaned_min.txt', 'SkipGram')
    w2v.train()



