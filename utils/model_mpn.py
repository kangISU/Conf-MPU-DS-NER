import torch
import numpy as np
from utils.model_pu import PU


class MPN(PU):
    def __init__(self, dp, class_num, inputSize=150):
        super(MPN, self).__init__(dp, class_num, inputSize=inputSize)

    def train_mini_batch(self, batch, args, optimizer):
        token, case, char, feature, label, flag = batch
        flag = self.padding_batch(flag)

        mask = [self.mask_of_flag(flag, i) for i in range(self.class_num)]

        optimizer.zero_grad()

        result = self.forward(token, case, char, feature)
        result_set = [result.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.class_num)
                      for i in range(self.class_num)]

        if args.weights == '':
            weights = [1 for i in range(1, self.class_num)]
        else:
            weights = [float(w) for w in args.weights.split(',')]

        neg_prior = 1 - sum(args.priors)

        # risk1 = P(+)_risk
        risk1 = sum([weights[i - 1] * args.priors[i - 1] * self.MAE(result_set[i], np.eye(self.class_num)[i])
                     for i in range(1, self.class_num)])

        # risk2 = N(-)_risk
        risk2 = neg_prior * self.MAE(result_set[0], np.eye(self.class_num)[0])

        risk = risk1 * args.m + risk2

        risk.backward()
        optimizer.step()

        return risk.item()

