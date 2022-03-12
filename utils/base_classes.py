import torch
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod
from progressbar import ProgressBar


class AbstractDataProcess(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset):
        self.dataset = dataset
        pass

    @abstractmethod
    def load_dataset(self, datasetName, flag, percent, suffix, no_lexicon):
        pass

    @staticmethod
    def get_casing(word, caseLookup):
        casing = 'other'
        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():
            casing = 'numeric'
        elif digitFraction > .5:
            casing = 'mainly_numeric'
        elif word.islower():
            casing = 'allLower'
        elif word.isupper():
            casing = 'allUpper'
        elif word[0].isupper():
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return caseLookup[casing]


class AbstractPU(nn.Module):
    def __init__(self, class_num, dim, inputSize=150, hiddenSize=200, layerNum=1, dropout=0.5):
        super(AbstractPU, self).__init__()
        self.class_num = class_num
        self.lstm = nn.LSTM(inputSize, hiddenSize, num_layers=layerNum, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hiddenSize, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, self.class_num),
        )
        self.softmax = nn.Softmax(dim=dim)

    @abstractmethod
    def forward(self, token, case, char, feature):
        pass

    @abstractmethod
    def train_mini_batch(self, batch, args, optimizer):
        pass


class Trainer(object):
    def __init__(self, model, lr):
        self.model = model
        self.saved_models = None
        self.learningRate = lr
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learningRate,
                                          weight_decay=1e-8)
        self.bestResult = 0

    def save(self, directory):
        if directory is not None:
            torch.save(self.model.state_dict(), directory)

    def decay_learning_rate(self, epoch, init_lr):
        lr = init_lr / (1 + 0.05 * epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def train(self, trainSet, validSet, dp, args):
        bar = ProgressBar(maxval=int((len(trainSet) - 1) / args.batch_size))
        epochs = args.epochs
        time = 0
        for e in range(1, epochs):
            print("\nEpoch: {}".format(e))
            bar.start()
            risks = []

            self.model.train()

            for step, batch in enumerate(dp.iterateSet(trainSet, batchSize=args.batch_size, mode="TRAIN", shuffle=True)):
                bar.update(step)
                risk = self.model.train_mini_batch(batch, args, self.optimizer)
                risks.append(risk)

            meanRisk = np.mean(np.array(risks))

            print("\nrisk on training set: {}".format(meanRisk))

            if e % 5 == 0:
                self.decay_learning_rate(e, args.lr)

            val_risk, f1_score = self.performance_on_dataset(validSet, dp, args, "valid")

            if args.early_stop:
                if f1_score <= self.bestResult:
                    time += 1
                else:
                    self.bestResult = f1_score
                    time = 0
                    self.save("saved_model/" + args.model_name)
                if time > 10 or (time > 1 and args.determine_entity):
                    print("\nBEST RESULT ON VALIDATE DATA: {}\n".format(self.bestResult))
                    break
        if not args.early_stop:
            self.save("saved_model/" + args.model_name)

    def test(self, batch, lengths):
        token, case, char, feature, labels = batch
        maxLen = max([x for x in lengths])
        mask = np.zeros((len(token), maxLen, self.model.class_num))  # num_sentences * max_sentence_length * class_num
        for i, x in enumerate(lengths):
            mask[i][:x][:] = 1
        self.model.eval()
        with torch.no_grad():
            result = self.model(token, case, char, feature)
            val_risk = self.model.risk_on_val(batch)
            result = result.masked_select(torch.from_numpy(mask).bool().cuda()).contiguous().view(-1, self.model.class_num)
            pred = torch.argmax(result, dim=1)

        prob = result[:, 1]

        return val_risk, pred.cpu().numpy(), prob.detach().cpu().numpy()

    def performance_on_dataset(self, dataset, dp, args, dataset_type, inference=False):
        pred = []
        corr = []
        risk = []
        words, efs = dp.words_efs_of_sentences(args, dataset_type)
        for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch) \
                in enumerate(dp.iterateSet(dataset, batchSize=100, mode="TEST", shuffle=False)):
            batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch]
            correcLabels = []
            for s in y_batch:
                for yi in s:
                    correcLabels.append(yi)
            lengths = [len(s) for s in x_word_batch]

            if inference:
                predLabels = self.bpu_inference_pred(batch, lengths)
            else:
                val_risk, predLabels, _ = self.test(batch, lengths)
                risk.append(val_risk)

            correcLabels = np.array(correcLabels)

            assert len(predLabels) == len(correcLabels)

            start = 0
            for i, l in enumerate(lengths):
                end = start + l
                p = predLabels[start:end]
                c = correcLabels[start:end]
                pred.append(p)
                corr.append(c)
                start = end

        meanRisk = np.mean(np.array(risk))
        print("risk on validation set: {}".format(meanRisk))
        print('Evaluation on Validation Set:')

        newSentences = []
        for i, s in enumerate(words):
            sent = []
            assert len(s) == len(efs[i]) == len(pred[i])
            for j, item in enumerate(s):
                sent.append([item, efs[i][j], pred[i][j]])
            newSentences.append(sent)

        trueEntityID, predEntityID = dp.entity_id_generation(newSentences)

        f1_record = []
        if args.determine_entity:
            labels = []
            preds = []
            for sent in newSentences:
                for token_info in sent:
                    labels.append(token_info[1])
                    preds.append(token_info[2])
            assert len(labels) == len(preds)
            p, r, f1 = dp.compute_token_f1(labels, preds)
            f1_record.append(f1)
            print("Entity: Precision: {}, Recall: {}, F1: {}".format(p, r, f1))
        else:
            if args.flag == 'ALL' or args.inference:
                flags = [f for f in args.tag2Idx.keys()][1:]
                for flag in flags:
                    precision, recall, f1 = dp.compute_precision_recall_f1(trueEntityID, predEntityID, flag, args.tag2Idx[flag])
                    print(flag + ": Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
                overall_precision, overall_recall, overall_f1 = dp.compute_overall_precision_recall_f1(trueEntityID, predEntityID)
                f1_record.append(overall_f1)
                print("OVERALL: Precision: {}, Recall: {}, F1: {}".format(overall_precision, overall_recall, overall_f1))
            else:
                p, r, f1 = dp.compute_precision_recall_f1(trueEntityID, predEntityID, args.flag, 1)
                f1_record.append(f1)
                print(args.flag + ": Precision: {}, Recall: {}, F1: {} on {}".format(p, r, f1, dataset_type))

        return meanRisk, sum(f1_record)

    def performance_on_testset(self, dataset, dp, args, dataset_type, model_path, pred_file):
        pred = []
        corr = []
        words, efs = dp.words_efs_of_sentences(args, dataset_type)
        for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch) \
                in enumerate(dp.iterateSet(dataset, batchSize=100, mode="TEST", shuffle=False)):
            batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch]
            correcLabels = []
            for s in y_batch:
                for yi in s:
                    correcLabels.append(yi)
            lengths = [len(s) for s in x_word_batch]

            self.model.load_state_dict(torch.load(model_path))
            _, predLabels, _ = self.test(batch, lengths)

            correcLabels = np.array(correcLabels)

            assert len(predLabels) == len(correcLabels)

            start = 0
            for i, l in enumerate(lengths):
                end = start + l
                p = predLabels[start:end]
                c = correcLabels[start:end]
                pred.append(p)
                corr.append(c)
                start = end

        newSentences = []
        for i, s in enumerate(words):
            sent = []
            assert len(s) == len(efs[i]) == len(pred[i])
            for j, item in enumerate(s):
                sent.append([item, efs[i][j], pred[i][j]])
            newSentences.append(sent)

        with open(pred_file, 'w', encoding='utf-8') as PRED:
            for sentence in newSentences:
                for word in sentence:
                    PRED.writelines(word[0] + ' ' + word[1] + ' ' + str(word[2]) + '\n')
                PRED.writelines('\n')

        trueEntityID, predEntityID = dp.entity_id_generation(newSentences)

        print("\nPERFORMANCE ON TEST DATA:\n")
        f1_record = []
        if args.determine_entity:
            labels = []
            preds = []
            for sent in newSentences:
                for token_info in sent:
                    labels.append(token_info[1])
                    preds.append(token_info[2])
            assert len(labels) == len(preds)
            p, r, f1 = dp.compute_token_f1(labels, preds)
            f1_record.append(f1)
            print("Entity: Precision: {}, Recall: {}, F1: {}".format(p, r, f1))
        else:
            if args.flag == 'ALL' or args.inference:
                flags = [f for f in args.tag2Idx.keys()][1:]
                for flag in flags:
                    precision, recall, f1 = dp.compute_precision_recall_f1(trueEntityID, predEntityID, flag, args.tag2Idx[flag])
                    print(flag + ": Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
                overall_precision, overall_recall, overall_f1 = dp.compute_overall_precision_recall_f1(trueEntityID, predEntityID)
                f1_record.append(overall_f1)
                print("OVERALL: Precision: {}, Recall: {}, F1: {}".format(overall_precision, overall_recall, overall_f1))
            else:
                p, r, f1 = dp.compute_precision_recall_f1(trueEntityID, predEntityID, args.flag, 1)
                f1_record.append(f1)
                print(args.flag + ": Precision: {}, Recall: {}, F1: {} on {}".format(p, r, f1, dataset_type))

        return sum(f1_record)

    def bpu_inference_pred(self, batch, lengths):
        pred_result = []
        for m in self.saved_models:
            self.model.load_state_dict(torch.load(m))
            _, predLabels, prob = self.test(batch, lengths)
            pred_result.append(prob)

        pred_result = np.array(pred_result).T
        res = []
        for pred in pred_result:
            if np.all(pred < 0.5):
                res.append(0)
            else:
                res.append(np.argmax(pred) + 1)
        return res

    def add_probs(self, dp, testset, model_path, datasetName, flag, suffix):
        self.model.load_state_dict(torch.load(model_path))
        probs = np.array([])
        First = True
        for step, (x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch) \
                in enumerate(dp.iterateSet(testset, batchSize=100, mode="TEST", shuffle=False)):
            batch = [x_word_batch, x_case_batch, x_char_batch, x_feature_batch, y_batch]
            lengths = [len(s) for s in x_word_batch]
            _, _, prob = self.test(batch, lengths)
            if First:
                probs = prob
                First = False
            else:
                probs = np.concatenate((probs, prob), axis=0)

        with open('data/' + datasetName + '/train.' + flag + '.txt', 'r', encoding='utf-8') as ORI, \
                open('data/' + datasetName + '/train.' + flag + '.txt.' + suffix, 'w', encoding='utf-8') as NEW:
            count = -1
            for line in ORI.readlines():
                if len(line.strip('\n')) > 0:
                    count += 1
                    NEW.writelines(line.strip('\n') + ' ' + str(probs[count]) + '\n')
                else:
                    NEW.writelines(line)

