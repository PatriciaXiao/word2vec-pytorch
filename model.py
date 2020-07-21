import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        X = torch.mean(self.embeddings(inputs), dim=0).view((1, -1))
        X = self.linear(X)
        X = F.log_softmax(X)
        return X

class BasicEncoder(nn.Module):
    '''
    reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        Note: encoder and decoder should be optimized separately in this case
    '''
    def __init__(self, vocab_size, embd_size, polarity_dim=1):
        super(BasicEncoder, self).__init__()
        # we should train separate embeddings for context and target, in the sense that one should not be assumed to appear in its own context
        self.context_embedding = nn.Embedding(vocab_size, embd_size)
        self.targets_embedding = nn.Embedding(vocab_size, embd_size)
        self.embedding_dim = embd_size
        self.polarity_dim = polarity_dim
        self.normalize_init(embd_size)
        # sizes
        self.d_n = embd_size - polarity_dim
        self.d_p = polarity_dim
        self.d = embd_size

    def normalize_init(self, embd_size):
        nn.init.normal_(self.context_embedding.weight, std=1.0 / math.sqrt(embd_size))
        nn.init.normal_(self.targets_embedding.weight, std=1.0 / math.sqrt(embd_size))

    def forward(self, inputs, target=True, part="all"):
        '''
        target or context, either embedding should be fine
        '''
        embedding_func = self.targets if target else self.context
        embedded = embedding_func(inputs)
        if part == "neutral":
            embedded = embedded[:,:,:-self.polarity_dim]
        elif part == "polarity":
            embedded = embedded[:,:,-self.polarity_dim:]
        return embedded

    def context(self, inputs):
        embedded = self.context_embedding(inputs)
        return embedded
    def targets(self, inputs):
        embedded = self.targets_embedding(inputs)
        return embedded

    def get_embedding(self, target=True):
        return self.targets_embedding if target else self.context_embedding

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = BasicEncoder(vocab_size, embedding_dim)
        self.u_embeddings = self.embeddings.targets
        self.v_embeddings = self.embeddings.context
        self.embedding_dim = embedding_dim

    def forward(self, u_pos, v_pos, v_neg):
        n_sample_pairs = pos_embed_v.shape[0]
        embed_u = self.u_embeddings(u_pos)
        pos_embed_v = self.v_embeddings(v_pos)
        pos_score = torch.sum(torch.mul(embed_u, pos_embed_v), dim = 1)
        pos_output = F.logsigmoid(pos_score).squeeze()

        neg_embed_v = self.v_embeddings(v_neg)
        neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim = -1)
        neg_output = F.logsigmoid(-1*neg_score).squeeze() #1-sigma(x)=sigma(-x)

        cost = pos_output + neg_output
        return -1 * cost.sum() / n_sample_pairs


    def save_embeddings(self, id2word, file_name, use_cuda):
        if use_cuda:
            embedding = self.embeddings.get_embedding().weight.cpu().data.numpy()
        else:
            embedding = self.embeddings.get_embedding().weight.data.numpy()

        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.embedding_dim))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
