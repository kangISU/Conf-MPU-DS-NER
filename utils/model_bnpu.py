import torch
import numpy as np

from utils.model_pu import PU


class BNPU(PU):
    def __init__(self, dp, class_num, inputSize=150):
        super(BNPU, self).__init__(dp, class_num, inputSize=inputSize)

    @staticmethod
    def loss_func(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        loss = torch.mean((y * (1 - yPred)).sum(dim=1))
        return loss

    def train_mini_batch(self, batch, args, optimizer):
        token, case, char, feature, label, flag = batch
        flag = self.padding_batch(flag)

        mask = [self.mask_of_flag(flag, i) for i in range(self.class_num)]

        optimizer.zero_grad()

        result = self.forward(token, case, char, feature)
        result_set = [result.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.class_num)
                      for i in range(self.class_num)]

        # risk1 = U(-)_risk - P(-)_risk
        risk1 = (self.loss_func(result_set[0], np.eye(self.class_num)[0]) -
                 sum([args.priors[i - 1] * self.loss_func(result_set[i], np.eye(self.class_num)[0]) for i in range(1, self.class_num)]))

        # risk2 = P(+)_risk
        risk2 = sum([args.priors[i - 1] * self.loss_func(result_set[i], np.eye(self.class_num)[i])
                     for i in range(1, self.class_num)])

        risk = risk1 + args.m * risk2

        if risk1 < args.beta:
            risk = - args.gamma * risk1

        risk.backward()
        optimizer.step()

        return risk.item()
