import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Flatten(nn.Module):
    def __init__(self, shape):
        super(Flatten, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(-1, self.shape)


class TimeDistributed(nn.Module):
    def __init__(self, module, char2Idx):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.char2Idx = char2Idx

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        if len(ids.size()) <= 2:
            return self.module(x)
        t, n = ids.size(0), ids.size(1)
        x_reshape = ids.contiguous().view(t * n, ids.size(2))
        y = self.module(x_reshape)
        y = y.contiguous().view(t, n, y.size()[1])
        return y, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            charID = []
            for cid in s:
                temp = []
                for id in cid:
                    temp.append(id)
                charID.append(temp)
            padding_vector = [self.char2Idx["PADDING"] for i in range(52)]
            charID += [padding_vector for _ in range(maxLength - len(charID))]
            ids.append(charID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


class CharCNN(nn.Module):
    def __init__(self, char2Idx):
        super(CharCNN, self).__init__()
        self.char2Idx = char2Idx
        self.embedding = nn.Embedding(len(self.char2Idx), 30)  # b*52*30
        self.embedding.weight.data.uniform_(-0.5, 0.5)
        self.dropout1 = nn.Dropout(0.5)
        self.conv1 = nn.Sequential(
            nn.Conv1d(30, 30, 3, 1, 1),  # b*30*52
            nn.Tanh(),
            nn.MaxPool1d(52),  # b*30*1
            Flatten(30)
        )
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.embedding(x)
        dropout = self.dropout1(embedding)
        dropout = dropout.permute(0, 2, 1)
        covout = self.conv1(dropout)
        res = self.dropout2(covout)
        return res


class CaseNet(nn.Module):
    def __init__(self, caseEmbeddings, case2Idx):
        super(CaseNet, self).__init__()
        self.caseEmbeddings = caseEmbeddings
        self.case2Idx = case2Idx
        self.embedding = nn.Embedding(caseEmbeddings.shape[0], self.caseEmbeddings.shape[1])
        self.embedding.weight.data.copy_(torch.from_numpy(self.caseEmbeddings))
        self.embedding.weight.requires_grad = False

        # self.dense = nn.Linear(self.caseEmbeddings.shape[1], self.caseEmbeddings.shape[1])

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        embeddings = self.embedding(ids)
        # embeddings = self.dense(embeddings)
        return embeddings, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            caseID = []
            for id in s:
                caseID.append(id)
            caseID += [self.case2Idx["PADDING_TOKEN"] for _ in range(maxLength - len(caseID))]
            ids.append(caseID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


class WordNet(nn.Module):
    def __init__(self, wordEmbeddings, word2Idx):
        super(WordNet, self).__init__()
        self.wordEmbeddings = wordEmbeddings
        self.word2Idx = word2Idx

        self.embedding = nn.Embedding(self.wordEmbeddings.shape[0], self.wordEmbeddings.shape[1])
        if wordEmbeddings is None:
            self.embedding.weight.data.normal_(0, 0.01)
        else:
            self.embedding.weight.data.copy_(torch.from_numpy(self.wordEmbeddings))
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        embeddings = self.embedding(ids)
        return embeddings, sortedLen, reversedIndices

    def embedding_with_padding(self, x, maxLength, length):
        ids = []
        for s in x:
            sentenceID = []
            for id in s:
                sentenceID.append(id)
            sentenceID += [self.word2Idx["PADDING_TOKEN"] for _ in range(maxLength - len(sentenceID))]
            ids.append(sentenceID)
        ids = Variable(torch.LongTensor(ids))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        ids = ids[indices]
        return ids.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

    def forward(self, x):
        length = [len(xi) for xi in x]
        maxLength = max(length)
        ids, sortedLen, reversedIndices = self.embedding_with_padding(x, maxLength, length)
        return ids, sortedLen, reversedIndices

    def embedding_with_padding(self, feature, maxLength, length):
        feature_ = []
        for sf in feature:
            f = []
            for wf in sf:
                f.append(wf)
            pad = np.zeros(12, dtype=int).tolist()
            f += [pad for _ in range(maxLength - len(f))]
            feature_.append(f)
        feature_ = Variable(torch.LongTensor(feature_))
        lengths = Variable(torch.LongTensor(length))
        sortedLen, indices = torch.sort(lengths, 0, descending=True)
        _, reversedIndices = torch.sort(indices, 0)
        feature_ = feature_[indices]
        return feature_.cuda(), sortedLen.data.numpy().tolist(), reversedIndices.cuda()
