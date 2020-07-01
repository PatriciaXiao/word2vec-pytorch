import time

import torch
from torch.autograd import Variable
import torch.optim as optim

from model import SkipGram, CBOW
from dataset import Dataset

import random
import numpy as np


class word2vec:
    def __init__(self, input_file, model_name, vocabulary_size=100000,
                 embedding_dim=200, epoch=10, batch_size=256, windows_size=5, neg_sample_size=10):
        self.model_name = model_name
        self.data = Dataset(input_file, vocabulary_size)
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.neg_sample_size = neg_sample_size

    def train(self):
        if self.model_name == 'SkipGram':
            model = SkipGram(self.vocabulary_size, self.embedding_dim)
        elif self.model_name == 'CBOW':
            return

        if torch.cuda.is_available():
            model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.2)

        for epoch in range(self.epoch):
            start = time.time()
            self.data.process =True
            batch_num = 0
            batch_new = 0

            while self.data.process:
                pos_u, pos_v, neg_v = self.data.generate_batch(self.windows_size, self.batch_size, self.neg_sample_size)

                '''
                print(pos_u.shape) # (2560,)
                print(pos_v.shape) # (2560,)
                print(neg_v.shape) # (2560, 10)
                exit(0)
                '''

                '''
                target = np.concatenate([pos_u.copy() for _ in range(neg_v.shape[1] + 1)])
                contex = np.concatenate([pos_v] + [neg_v[:,i] for i in range(neg_v.shape[1])])
                labels = np.array([1 for _ in pos_v] + [-1 for _ in range(neg_v.shape[0] * neg_v.shape[1])])
                target = torch.LongTensor(target)
                contex = torch.LongTensor(contex)
                labels = torch.LongTensor(labels)
                '''

                pos_u = torch.LongTensor(pos_u)
                pos_v = torch.LongTensor(pos_v)
                neg_v = torch.LongTensor(neg_v)

                if torch.cuda.is_available():
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, self.batch_size)
                # loss = model(pos_u, pos_v, neg_v, self.batch_size, target, contex, labels)
                loss.backward()
                optimizer.step()

                if batch_num % 3000 == 0:
                    end = time.time()
                    print('epoch,batch = %2d %5d:   batch_size = %5d  loss = %4.3f\r'
                          % (epoch, batch_num, self.batch_size, loss.item()), end="\n")
                    batch_new = batch_num
                    start = time.time()
                batch_num += 1

        model.save_embeddings(self.data.idx2word, 'word_embdding.txt', torch.cuda.is_available())


if __name__ == '__main__':

    fixed_seed = 1
    if fixed_seed is not None:
        torch.manual_seed(fixed_seed)
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        torch.cuda.manual_seed(fixed_seed)

    # w2v = word2vec('text8', 'SkipGram')
    w2v = word2vec('toy', 'SkipGram')
    # w2v = word2vec('democratic_cleaned_min.txt', 'SkipGram')
    w2v.train()



