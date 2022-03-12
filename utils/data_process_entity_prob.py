from utils.data_process import DataProcess
import numpy as np


class DataProcessEntityProb(DataProcess):
    def __init__(self, args):
        super(DataProcessEntityProb, self).__init__(args)

    def read_processed_file(self, filename, flag):
        with open(filename, "r", encoding='utf-8') as fw:
            sentences = []
            sentence = []
            for indx, line in enumerate(fw):
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    splits = line.split(' ')
                    if len(splits[0].strip()) > 0:
                        if splits[1].strip() != "-1":
                            if "ALL" not in flag:
                                if "train" not in filename:
                                    sentence.append([splits[0].strip(), int(splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag),
                                                     int(splits[2])])
                                else:
                                    sentence.append([splits[0].strip(), int(splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag),
                                                     int(splits[2]), float(splits[3])])
                            else:
                                tflag = splits[1].strip().split("-")
                                if len(tflag) > 1:
                                    tflag = self.tag2Idx[tflag[1]]
                                else:
                                    tflag = 0
                                if "train" not in filename:
                                    sentence.append([splits[0].strip(), tflag, int(splits[2])])
                                else:
                                    sentence.append([splits[0].strip(), tflag, int(splits[2]), float(splits[3])])
                        else:
                            sentence.append([splits[0].strip(), -1, int(splits[2])])

                    else:
                        if splits[1].strip() != "-1":
                            if "ALL" not in flag:
                                if "train" not in filename:
                                    sentence.append([splits[0].strip(), int(splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag),
                                                     int(splits[2])])
                                else:
                                    sentence.append([splits[0].strip(), int(splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag),
                                                     int(splits[2]), float(splits[3])])
                            else:
                                tflag = splits[1].strip().split("-")
                                if len(tflag) > 1:
                                    tflag = self.tag2Idx[tflag[1]] + 1
                                else:
                                    tflag = 0
                                if "train" not in filename:
                                    sentence.append([splits[0].strip(), tflag, int(splits[2])])
                                else:
                                    sentence.append([splits[0].strip(), tflag, int(splits[2]), float(splits[3])])
                        else:
                            sentence.append([splits[0].strip(), -1, int(splits[2])])
            if len(sentence) > 0:
                sentences.append(sentence)

            return sentences

    def load_dataset(self, datasetName, flag, percent, suffix, no_lexicon):
        if suffix == "":
            fname = "data/" + datasetName + "/train." + flag + ".txt"
        else:
            fname = "data/" + datasetName + "/train." + flag + ".txt" + "." + suffix
        trainSentences = self.read_processed_file(fname, flag)
        trainSize = int(len(trainSentences) * percent)
        trainSentences = trainSentences[:trainSize]
        prob = [[w[-1] for w in sent] for sent in trainSentences]
        self.add_char_info(trainSentences)
        self.add_dict_info(trainSentences, 3, datasetName, no_lexicon)
        train_sentences_X, train_sentences_Y, train_sentences_LF = self.padding(
            self.createMatrices(trainSentences, self.word2Idx, self.case2Idx, self.char2Idx))

        validSentences = self.read_processed_file("data/" + datasetName + "/valid.txt", flag)
        self.add_char_info(validSentences)
        self.add_dict_info(validSentences, 3, datasetName, no_lexicon)
        valid_sentences_X, valid_sentences_Y, valid_sentences_LF = self.padding(
            self.createMatrices(validSentences, self.word2Idx, self.case2Idx, self.char2Idx))

        testSentences = self.read_processed_file("data/" + datasetName + "/test.txt", flag)
        self.add_char_info(testSentences)
        self.add_dict_info(testSentences, 3, datasetName, no_lexicon)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(testSentences, self.word2Idx, self.case2Idx, self.char2Idx))

        dataset = ((train_sentences_X, train_sentences_Y, train_sentences_LF),
                   (valid_sentences_X, valid_sentences_Y, valid_sentences_LF),
                   (test_sentences_X, test_sentences_Y, test_sentences_LF))

        trainSet, true_priors = self.make_PU_dataset(dataset)
        trainX, trainY, FG = zip(*trainSet)
        trainSet = list(zip(trainX, trainY, FG, prob))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, true_priors

    def iterateSet(self, trainset, batchSize, mode, shuffle=True):
        if mode == "TRAIN":
            data_size = len(trainset)
            X, Y, FG, PR = zip(*trainset)
            X = np.array(X, dtype=object)
            Y = np.array(Y, dtype=object)
            FG = np.array(FG, dtype=object)
            PR = np.array(PR, dtype=object)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                self.shuffle_indices = np.random.permutation(np.arange(data_size))
                x = X[self.shuffle_indices]
                y = Y[self.shuffle_indices]
                flag = FG[self.shuffle_indices]
                prob = PR[self.shuffle_indices]
            else:
                self.shuffle_indices = np.arange(data_size)
                x = X
                y = Y
                flag = FG
                prob = PR

            for batch_num in range(num_batches_per_epoch):
                self.start_index = batch_num * batchSize
                self.end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                features = []
                labels = []
                flags = []
                probs = []
                data_X = x[self.start_index:self.end_index]
                data_Y = y[self.start_index:self.end_index]
                data_FG = flag[self.start_index:self.end_index]
                data_PR = prob[self.start_index:self.end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                for dt in data_Y:
                    dt = np.array(dt)
                    dt = dt.astype('int32')
                    labels.append(dt)
                for dt in data_FG:
                    dt = np.array(dt)
                    dt = dt.astype('int32')
                    flags.append(dt)
                for dt in data_PR:
                    pr = []
                    for d in dt:
                        d = np.array(d)
                        d = d.astype('float')
                        pr.append(d)
                    probs.append(pr)

                yield np.asarray(tokens, dtype=object), np.asarray(caseing, dtype=object), \
                      np.asarray(char, dtype=object), np.asarray(features, dtype=object), \
                      np.asarray(labels, dtype=object), np.asarray(flags, dtype=object), \
                      np.asarray(probs, dtype=object)
        else:
            data_size = len(trainset)
            try:
                X, Y, _ = zip(*trainset)
            except:
                try:
                    X, Y, _, _ = zip(*trainset)
                except:
                    print("dataset error!")
            X = np.array(X, dtype=object)
            Y = np.array(Y, dtype=object)

            num_batches_per_epoch = int((len(trainset) - 1) / batchSize) + 1
            if shuffle:
                self.shuffle_indices = np.random.permutation(np.arange(data_size))
                x = np.array(X)[self.shuffle_indices]
                y = np.array(Y)[self.shuffle_indices]
            else:
                x = X
                y = Y
            for batch_num in range(num_batches_per_epoch):
                self.start_index = batch_num * batchSize
                self.end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                features = []
                data_X = x[self.start_index:self.end_index]
                data_Y = y[self.start_index:self.end_index]
                for dt in data_X:
                    t, c, ch, f = dt
                    tokens.append(t)
                    caseing.append(c)
                    char.append(ch)
                    features.append(f)
                yield np.asarray(tokens, dtype=object), np.asarray(caseing, dtype=object), \
                      np.asarray(char, dtype=object), np.asarray(features, dtype=object), \
                      np.asarray(data_Y, dtype=object)
