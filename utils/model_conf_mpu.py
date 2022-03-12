import torch
import numpy as np
from utils.model_pu import PU


class ConfMPU(PU):
    def __init__(self, dp, class_num, inputSize=150):
        super(ConfMPU, self).__init__(dp, class_num, inputSize=inputSize)

    @staticmethod
    def conf_MAE(yPred, yTrue, prob):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        prob = prob.float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean((temp * 1 / prob).sum(dim=1) / yTrue.shape[0])
        return loss

    def mask_of_flag_prob(self, eta, flag, prob, class_elem):
        f_masks = []
        p_masks = []
        for s_f, s_p in zip(flag, prob):
            s_mask_f = []
            s_mask_p = []
            for w_f, w_p in zip(s_f, s_p):
                if w_f == class_elem and w_f > 0 and w_p > eta:
                    s_mask_f.append([1] * self.class_num)
                    s_mask_p.append([w_p])
                elif w_f == class_elem and w_f == 0 and w_p <= eta:
                    s_mask_f.append([1] * self.class_num)
                    s_mask_p.append([w_p])
                else:
                    s_mask_f.append([0] * self.class_num)

            f_masks.append(s_mask_f)
            p_masks.append(s_mask_p)
        return np.array(f_masks), p_masks

    def train_mini_batch(self, batch, args, optimizer):
        token, case, char, feature, label, flag, prob = batch
        flag = self.padding_batch(flag)
        prob = self.padding_batch(prob)

        optimizer.zero_grad()

        f_mask = []
        p_mask = []
        eta = float(args.eta)
        for i in range(self.class_num):
            mask1, mask2 = self.mask_of_flag_prob(eta, flag, prob, i)
            f_mask.append(mask1)
            p_mask.append(mask2)

        mask = [self.mask_of_flag(flag, i) for i in range(self.class_num)]

        result = self.forward(token, case, char, feature)

        result_set = [result.masked_select(torch.from_numpy(f_mask[i]).bool().cuda()).contiguous().view(-1, self.class_num)
                      for i in range(self.class_num)]

        result_set2 = [result.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.class_num)
                       for i in range(self.class_num)]

        prob_set = [sum(ele, []) for ele in p_mask]
        prob_set = [torch.from_numpy(np.array(ele)) for ele in prob_set]

        if args.weights == '':
            weights = [1 for i in range(1, self.class_num)]
        else:
            weights = [float(w) for w in args.weights.split(',')]

        # U'(-)
        risk1 = self.MAE(result_set[0], np.eye(self.class_num)[0])
        # P'(-)
        risk2 = sum([weights[i - 1] * args.priors[i - 1] * self.conf_MAE(result_set[i], np.eye(self.class_num)[0], prob_set[i])
                     for i in range(1, self.class_num)])
        # P(-)
        risk3 = sum([weights[i - 1] * args.priors[i - 1] * self.MAE(result_set2[i], np.eye(self.class_num)[0])
                     for i in range(1, self.class_num)])
        # P(+)
        risk4 = sum([weights[i - 1] * args.priors[i - 1] * self.MAE(result_set2[i], np.eye(self.class_num)[i])
                     for i in range(1, self.class_num)])

        negative_risk = risk1
        positive_risk = risk2 - risk3 + risk4
        risk = positive_risk * args.m + negative_risk
        if positive_risk < 0:
            risk = negative_risk

        risk.backward()
        optimizer.step()

        return risk.item()
