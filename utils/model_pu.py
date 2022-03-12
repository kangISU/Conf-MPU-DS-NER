import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils.base_classes import AbstractPU
from sub_model import CharCNN, CaseNet, WordNet, FeatureNet, TimeDistributed


class PU(AbstractPU):
    def __init__(self, dp, class_num, inputSize=150):
        super(PU, self).__init__(class_num, 2, inputSize=inputSize)
        self.dp = dp
        self.charModel = TimeDistributed(CharCNN(self.dp.char2Idx), self.dp.char2Idx)
        self.wordModel = WordNet(self.dp.wordEmbeddings, self.dp.word2Idx)
        self.caseModel = CaseNet(self.dp.caseEmbeddings, self.dp.case2Idx)
        self.featureModel = FeatureNet()

        if torch.cuda.is_available:
            self.charModel.cuda()
            self.wordModel.cuda()
            self.caseModel.cuda()
            self.featureModel.cuda()

    def forward(self, token, case, char, feature):
        charOut, sortedLen1, reversedIndices1 = self.charModel(char)
        wordOut, sortedLen2, reversedIndices2 = self.wordModel(token)
        caseOut, sortedLen3, reversedIndices3 = self.caseModel(case)
        featureOut, sortedLen4, reversedIndices4 = self.featureModel(feature)

        encoding = torch.cat([wordOut.float(), caseOut.float(), charOut.float(), featureOut.float()], dim=2)

        sortedLen = sortedLen1
        reverseIndices = reversedIndices1

        packed_embeds = pack_padded_sequence(encoding, sortedLen, batch_first=True)

        maxLen = sortedLen[0]
        mask = torch.zeros([len(sortedLen), maxLen, self.class_num])
        for i, l in enumerate(sortedLen):
            mask[i][:l][:] = 1

        lstmOut, (h, _) = self.lstm(packed_embeds)

        paddedOut = pad_packed_sequence(lstmOut, batch_first=True)
        fcOut = self.fc(paddedOut[0])
        fcOut = self.softmax(fcOut)

        fcOut = fcOut * mask.cuda()
        fcOut = fcOut[reverseIndices]

        return fcOut

    def mask_of_flag(self, flag, class_elem):
        masks = []
        for s in flag:
            s_mask = []
            for w in s:
                if w == class_elem:
                    s_mask.append([1] * self.class_num)  # [1,1,1,1,1] if class_num = 5
                else:
                    s_mask.append([0] * self.class_num)
            masks.append(s_mask)
        return np.array(masks)

    @staticmethod
    def padding_batch(data):
        length = [len(i) for i in data]
        maxLen = max(length)
        rst = []
        for s in data:
            f = list(s)
            f += [-1 for _ in range(maxLen - len(f))]
            rst.append(f)
        return rst

    @staticmethod
    def SMAE(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        temp = 1 - yPred
        loss = torch.mean((y * temp).sum(dim=1))
        return loss

    @staticmethod
    def MAE(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean(temp.sum(dim=1) / yTrue.shape[0])
        return loss

    def train_mini_batch(self, batch, args, optimizer):
        pass

    def risk_on_val(self, batch):
        token, case, char, feature, label = batch
        flag = self.padding_batch(label)

        mask = [self.mask_of_flag(flag, i) for i in range(self.class_num)]

        result = self.forward(token, case, char, feature)
        result_set = [result.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.class_num)
                      for i in range(self.class_num)]

        risk = sum([self.MAE(result_set[i], np.eye(self.class_num)[i]) for i in range(0, self.class_num)])

        return risk.item()

