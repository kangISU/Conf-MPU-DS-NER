from utils.base_classes import AbstractDataProcess
import numpy as np
import pickle


class DataProcess(AbstractDataProcess):
    def __init__(self, args):
        super(DataProcess, self).__init__(args.dataset)
        self.shuffle_indices = []
        self.start_index = 0
        self.end_index = 0

        self.tag2Idx = args.tag2Idx
        self.idx2tag = args.idx2tag
        self.case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                         'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(self.case2Idx), dtype='float32')
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            self.char2Idx[c] = len(self.char2Idx)
        self.words = self.get_words(args.dataset)
        self.word2Idx = {}
        self.wordEmbeddings = []
        if args.embedding == "glove":
            self.get_embedding("data/glove.6B.100d.txt")
        elif args.embedding == "bio-embedding":
            self.get_embedding("data/bio_embedding.txt")
        elif args.embedding == "word2vec":
            self.get_embedding("data/word2vec_100_skip.txt")
        else:
            self.get_embedding("data/glove.6B.100d.txt")
        self.dataset = args.dataset
        self.class_num = args.cn
        self.priors = args.priors

    def get_embedding(self, embedding_file):
        with open(embedding_file, "r", encoding='utf-8') as fw:
            for line in fw:
                line = line.strip()
                splits = line.split(" ")

                if len(self.word2Idx) == 0:
                    self.word2Idx["PADDING_TOKEN"] = len(self.word2Idx)
                    vector = np.zeros(len(splits) - 1)  # Zero vector vor 'PADDING' word
                    self.wordEmbeddings.append(vector)

                    self.word2Idx["UNKNOWN_TOKEN"] = len(self.word2Idx)
                    vector = np.random.uniform(-0.25, 0.25, len(splits) - 1)
                    self.wordEmbeddings.append(vector)

                if splits[0].lower() in self.words:
                    vector = np.array([float(num) for num in splits[1:]])
                    self.wordEmbeddings.append(vector)
                    self.word2Idx[splits[0]] = len(self.word2Idx)
        self.wordEmbeddings = np.array(self.wordEmbeddings)

    def get_words(self, dataset):
        words = {}
        trainSentences = self.read_origin_file("data/" + dataset + "/train.txt")
        validSentences = self.read_origin_file("data/" + dataset + "/valid.txt")
        testSentences = self.read_origin_file("data/" + dataset + "/test.txt")
        for sentences in [trainSentences, validSentences, testSentences]:
            for sentence in sentences:
                for token, label, flag in sentence:
                    words[token.lower()] = True
        return words

    def add_dict_info(self, sentences, windowSize, datasetName, no_lexicon):
        if no_lexicon:
            for i, sentence in enumerate(sentences):
                for j, data in enumerate(sentence):
                    feature = np.zeros([4, windowSize], dtype=int)
                    feature = feature.reshape([-1]).tolist()
                    sentences[i][j] = [data[0], data[1], feature, data[2], data[3]]
            print('***no lexicon feature***')
            return

        if "CoNLL2003" in datasetName:
            perBigDic = set()
            locBigDic = set()
            orgBigDic = set()
            miscBigDic = set()
            with open("dictionaries/" + datasetName + "/personBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        perBigDic.add(line)
            with open("dictionaries/" + datasetName + "/locationBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        locBigDic.add(line)
            with open("dictionaries/" + datasetName + "/organizationBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        orgBigDic.add(line)
            with open("dictionaries/" + datasetName + "/miscBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        miscBigDic.add(line)

            for i, sentence in enumerate(sentences):
                for j, data in enumerate(sentence):
                    feature = np.zeros([4, windowSize], dtype=int)
                    maxLen = len(sentence)
                    remainLenRight = maxLen - j - 1
                    rightSize = min(remainLenRight, windowSize - 1)
                    remainLenLeft = j
                    leftSize = min(remainLenLeft, windowSize - 1)
                    k = 0
                    words = [sentence[j][0]]

                    while k < rightSize:
                        # right side
                        word = sentence[j + k + 1][0]
                        temp = words[-1]
                        word = temp + " " + word
                        words.append(word)
                        k += 1

                    k = 0
                    while k < leftSize:
                        # left side
                        word = sentence[j - k - 1][0]
                        temp = words[0]
                        word = word + " " + temp
                        words.insert(0, word)
                        k += 1

                    for idx, word in enumerate(words):
                        count = len(word.split())
                        if word in perBigDic:
                            feature[self.tag2Idx["PER"] - 1][count - 1] = 1
                        elif word in locBigDic:
                            feature[self.tag2Idx["LOC"] - 1][count - 1] = 1
                        elif word in orgBigDic:
                            feature[self.tag2Idx["ORG"] - 1][count - 1] = 1
                        elif word in miscBigDic:
                            feature[self.tag2Idx["MISC"] - 1][count - 1] = 1
                    feature = feature.reshape([-1]).tolist()
                    sentences[i][j] = [data[0], data[1], feature, data[2], data[3]]
        elif 'BC5CDR' in datasetName:
            chemicalBigDic = set()
            diseaseBigDic = set()
            with open("dictionaries/" + datasetName + "/chemicalBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        chemicalBigDic.add(line)
            with open("dictionaries/" + datasetName + "/diseaseBigDic.txt", "r", encoding='utf-8') as fw:
                for line in fw:
                    line = line.strip()
                    if len(line) > 0:
                        diseaseBigDic.add(line)

            for i, sentence in enumerate(sentences):
                for j, data in enumerate(sentence):
                    feature = np.zeros([4, windowSize], dtype=int)
                    maxLen = len(sentence)
                    remainLenRight = maxLen - j - 1
                    rightSize = min(remainLenRight, windowSize - 1)
                    remainLenLeft = j
                    leftSize = min(remainLenLeft, windowSize - 1)
                    k = 0
                    words = [sentence[j][0]]

                    while k < rightSize:
                        # right side
                        word = sentence[j + k + 1][0]
                        temp = words[-1]
                        word = temp + " " + word
                        words.append(word)
                        k += 1

                    k = 0
                    while k < leftSize:
                        # left side
                        word = sentence[j - k - 1][0]
                        temp = words[0]
                        word = word + " " + temp
                        words.insert(0, word)
                        k += 1

                    for idx, word in enumerate(words):
                        count = len(word.split())
                        if word in chemicalBigDic:
                            feature[self.tag2Idx["Chemical"] - 1][count - 1] = 1
                        elif word in diseaseBigDic:
                            feature[self.tag2Idx["Disease"] - 1][count - 1] = 1
                    feature = feature.reshape([-1]).tolist()
                    sentences[i][j] = [data[0], data[1], feature, data[2], data[3]]
        else:
            for i, sentence in enumerate(sentences):
                for j, data in enumerate(sentence):
                    feature = np.zeros([4, windowSize], dtype=int)
                    feature = feature.reshape([-1]).tolist()
                    sentences[i][j] = [data[0], data[1], feature, data[2], data[3]]

    @staticmethod
    def add_char_info(sentences):
        for i, sentence in enumerate(sentences):
            for j, data in enumerate(sentence):
                chars = [c for c in data[0]]
                sentences[i][j] = [data[0], chars, data[1], data[2]]

    def createMatrices(self, sentences, word2Idx, case2Idx, char2Idx):
        unknownIdx = word2Idx['UNKNOWN_TOKEN']
        paddingIdx = word2Idx['PADDING_TOKEN']

        dataset = []

        wordCount = 0
        unknownWordCount = 0

        for sentence in sentences:
            wordIndices = []
            caseIndices = []
            charIndices = []
            featureList = []
            entityFlags = []
            labeledFlags = []

            for word, char, feature, ef, lf in sentence:
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                charIdx = []
                for x in char:
                    if x in char2Idx:
                        charIdx.append(char2Idx[x])
                    else:
                        charIdx.append(char2Idx["UNKNOWN"])

                wordIndices.append(wordIdx)
                caseIndices.append(self.get_casing(word, case2Idx))
                charIndices.append(charIdx)
                featureList.append(feature)
                entityFlags.append(ef)
                labeledFlags.append(lf)

            dataset.append([wordIndices, caseIndices, charIndices, featureList, entityFlags, labeledFlags])
        return dataset

    @staticmethod
    def padding(sentences):
        maxlen = 52
        for i, sentence in enumerate(sentences):
            mask = np.zeros([len(sentences[i][2]), maxlen])
            for j, chars in enumerate(sentences[i][2]):
                for k, c in enumerate(chars):
                    if k < maxlen:
                        mask[j][k] = c
            sentences[i][2] = mask.tolist()

        sentences_X = []
        sentences_Y = []
        sentences_LF = []

        for i, sentence in enumerate(sentences):
            sentences_X.append(sentence[:4])
            sentences_Y.append(sentence[4])
            sentences_LF.append(sentence[5])
        return np.array(sentences_X, dtype=object), \
               np.array(sentences_Y, dtype=object), \
               np.array(sentences_LF, dtype=object)

    def make_PU_dataset(self, dataset):

        def _make_PU_dataset(x, y, flag):
            n_flag = {}
            for i in range(1, self.class_num):
                n_flag[i] = 0
            all_item = 0
            for item in y:
                item = np.array(item)
                for i in range(1, self.class_num):
                    n_flag[i] += (item == i).sum()
                all_item += len(item)

            priors = [float(n_flag[i]) / float(all_item) for i in range(1, self.class_num)]  # cannot do this in real data
            print("true prior(s): " + str(priors))
            # CoNLL2003: [0.05465055176037835, 0.040747270664617107, 0.04923362521547384, 0.02255661253014178]
            print("estimated prior(s): " + str(self.priors))
            return x, y, flag, priors

        (_train_X, _train_Y, _labeledFlag), (_, _, _), (_, _, _) = dataset
        X, Y, FG, priors = _make_PU_dataset(_train_X, _train_Y, _labeledFlag)

        return list(zip(X, Y, FG)), priors

    @staticmethod
    def entity_id_generation(sentences):
        sent_id = 0
        type_ = "#"
        flag = -1

        label_start_id = 0
        pred_start_id = 0

        true_entities = []
        pred_entities = []
        for sentence in sentences:
            pre_label = "O"
            sent_true_entities = []
            sent_pred_entities = []
            for i, (word, label, pred) in enumerate(sentence):
                if label == "O":
                    if not pre_label == "O":
                        label_end_id = i - 1
                        sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))
                else:
                    if "B-" in label:
                        label = label.split("-")[-1]
                        if not pre_label == "O":
                            label_end_id = i - 1
                            sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))
                        label_start_id = i
                        type_ = label
                    else:
                        continue
                pre_label = label
            if not pre_label == "O":
                label_end_id = len(sentence) - 1
                sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))

            pre_pred = 0
            for i, (word, label, pred) in enumerate(sentence):
                if pred == 0:
                    if not pre_pred == 0:
                        pred_end_id = i - 1
                        sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
                else:
                    if not pre_pred == pred:
                        if not pre_pred == 0:
                            pred_end_id = i - 1
                            sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
                        pred_start_id = i
                        flag = pred
                    else:
                        continue
                pre_pred = pred

            if not pre_pred == 0:
                pred_end_id = len(sentence) - 1
                sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))

            sent_id += 1
            true_entities.append(sent_true_entities)
            pred_entities.append(sent_pred_entities)
        return true_entities, pred_entities

    @staticmethod
    def read_origin_file(filename):
        with open(filename, "r", encoding='utf-8') as fw:
            sentences = []
            sentence = []
            for line in fw:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    splits = line.split(' ')
                    sentence.append([splits[0].strip(), splits[1].strip(), np.zeros(5)])

            if len(sentence) > 0:
                sentences.append(sentence)

            return sentences

    def read_processed_file(self, filename, flag):
        with open(filename, "r", encoding='utf-8') as fw:
            sentences = []
            sentence = []
            for line in fw:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                else:
                    splits = line.split(' ')
                    if len(splits[0].strip()) > 0:
                        if splits[1].strip() != "-1":
                            if "Entity" in flag:
                                sentence.append([splits[0].strip(), int(
                                    splits[1].strip().startswith("B-") or splits[1].strip().startswith("I-")), int(splits[2])])
                            elif "ALL" in flag:
                                tflag = splits[1].strip().split("-")[-1]
                                if tflag in self.tag2Idx:
                                    tflag = self.tag2Idx[tflag]
                                else:
                                    tflag = 0
                                sentence.append([splits[0].strip(), tflag, int(splits[2])])
                            else:
                                sentence.append([splits[0].strip(), int(
                                    splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag), int(splits[2])])
                        else:
                            sentence.append([splits[0].strip(), -1, int(splits[2])])

                    else:
                        if splits[1].strip() != "-1":
                            if "Entity" in flag:
                                sentence.append([splits[0].strip(), int(
                                    splits[1].strip().startswith("B-") or splits[1].strip().startswith("I-")), int(splits[2])])
                            elif "ALL" in flag:
                                tflag = splits[1].strip().split("-")[-1]
                                if tflag in self.tag2Idx:
                                    tflag = self.tag2Idx[tflag]
                                else:
                                    tflag = 0
                                sentence.append([splits[0].strip(), tflag, int(splits[2])])
                            else:
                                sentence.append([splits[0].strip(), int(
                                    splits[1].strip() == "B-" + flag or splits[1].strip() == "I-" + flag), int(splits[2])])
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
        trainSet = list(zip(trainX, trainY, FG))
        validSet = list(zip(valid_sentences_X, valid_sentences_Y, valid_sentences_LF))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return trainSet, validSet, testSet, true_priors

    def load_testset(self, datasetName, file, no_lexicon):
        sentences = self.read_processed_file("data/" + datasetName + "/" + file, "")
        self.add_char_info(sentences)
        self.add_dict_info(sentences, 3, datasetName, no_lexicon)
        test_sentences_X, test_sentences_Y, test_sentences_LF = self.padding(
            self.createMatrices(sentences, self.word2Idx, self.case2Idx, self.char2Idx))
        testSet = list(zip(test_sentences_X, test_sentences_Y, test_sentences_LF))
        return testSet

    def iterateSet(self, dataset, batchSize, mode, shuffle=True):
        if mode == "TRAIN":
            data_size = len(dataset)
            X, Y, FG = zip(*dataset)
            X = np.array(X, dtype=object)
            Y = np.array(Y, dtype=object)
            FG = np.array(FG, dtype=object)

            num_batches_per_epoch = int((len(dataset) - 1) / batchSize) + 1
            if shuffle:
                self.shuffle_indices = np.random.permutation(np.arange(data_size))
                x = X[self.shuffle_indices]
                y = Y[self.shuffle_indices]
                flag = FG[self.shuffle_indices]
            else:
                self.shuffle_indices = np.arange(data_size)
                x = X
                y = Y
                flag = FG

            for batch_num in range(num_batches_per_epoch):
                self.start_index = batch_num * batchSize
                self.end_index = min((batch_num + 1) * batchSize, data_size)
                tokens = []
                caseing = []
                char = []
                features = []
                labels = []
                flags = []
                data_X = x[self.start_index:self.end_index]
                data_Y = y[self.start_index:self.end_index]
                data_FG = flag[self.start_index:self.end_index]

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

                yield np.asarray(tokens, dtype=object), np.asarray(caseing, dtype=object), \
                      np.asarray(char, dtype=object), np.asarray(features, dtype=object), \
                      np.asarray(labels, dtype=object), np.asarray(flags, dtype=object)
        else:
            data_size = len(dataset)
            try:
                X, Y, _ = zip(*dataset)
            except:
                try:
                    X, Y, _, _ = zip(*dataset)
                except:
                    print("dataset error!")
            X = np.array(X, dtype=object)
            Y = np.array(Y, dtype=object)

            num_batches_per_epoch = int((len(dataset) - 1) / batchSize) + 1
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

    @staticmethod
    def compute_token_f1(labels, preds):
        # recall = tp/(tp + fn)
        # precision = tp/(tp + fp)
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        assert len(labels) == len(preds)
        for i in range(len(labels)):
            if (labels[i].startswith("B") or labels[i].startswith("I")) and preds[i] == 1:
                tp += 1
            elif (labels[i].startswith("B") or labels[i].startswith("I")) and preds[i] == 0:
                fn += 1
            elif labels[i].startswith("O") and preds[i] == 0:
                tn += 1
            elif labels[i].startswith("O") and preds[i] == 1:
                fp += 1
        if tp == 0:
            recall = 0
            precision = 0
        else:
            recall = float(tp) / (float(tp) + float(fn))
            precision = float(tp) / (float(tp) + float(fp))
        if recall == 0 or precision == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute_precision_recall_f1(true_entities, pred_entities, flag, pflag):
        tp = 0
        np_ = 0
        pp = 0
        for i in range(len(true_entities)):
            sent_true = true_entities[i]
            sent_pred = pred_entities[i]
            for e in sent_true:
                if flag in e:
                    np_ += 1
                    temp = e.replace(flag, str(pflag))
                    if temp in sent_pred:
                        tp += 1
            for e in sent_pred:
                if int(e.split("_")[-1]) == pflag:
                    pp += 1
        if pp == 0:
            p = 0
        else:
            p = float(tp) / float(pp)
        if np_ == 0:
            r = 0
        else:
            r = float(tp) / float(np_)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = float(2 * p * r) / float((p + r))
        return p, r, f1

    def compute_overall_precision_recall_f1(self, true_entities, pred_entities):
        tp = 0
        np_ = len(sum(true_entities, []))
        pp = len(sum(pred_entities, []))
        temp = ' '

        assert len(true_entities) == len(pred_entities)
        for i in range(len(true_entities)):
            sent_true = true_entities[i]
            sent_pred = pred_entities[i]
            for e in sent_true:
                for flag in self.tag2Idx:
                    if flag in e:
                        temp = e.replace(flag, str(self.tag2Idx[flag]))
                if temp in sent_pred:
                    tp += 1
        if pp == 0:
            p = 0
        else:
            p = float(tp) / float(pp)
        if np_ == 0:
            r = 0
        else:
            r = float(tp) / float(np_)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = float(2 * p * r) / float((p + r))
        return p, r, f1

    def words_efs_of_sentences(self, args, dataset_type):
        data_sentences = self.read_origin_file("data/" + args.dataset + "/" + dataset_type + ".txt")
        dataSize = int(len(data_sentences) * args.pert)
        data_sentences = data_sentences[:dataSize]
        data_words = []
        data_efs = []
        for s in data_sentences:
            temp = []
            temp2 = []
            for word, ef, lf in s:
                temp.append(word)
                temp2.append(ef)
            data_words.append(temp)
            data_efs.append(temp2)
        return data_words, data_efs
