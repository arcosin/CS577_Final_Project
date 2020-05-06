

import csv
from collections import defaultdict
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd

# Gensim W2V imports.
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# SKLearn metric imports.
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Torch imports.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(13011)



#WORD_2_VEC_PATH = "/homes/cs577/hw2/w2v.bin"      # Purdue network location.
WORD_2_VEC_PATH = "D:\\Word2Vec\\w2v.bin"          # Max's computer location.



def readData(filename, experimental = True):
    df = pd.read_csv(filename)
    if experimental:
        df = df.loc[df['category'] != 'baseline']
    else:
        df = df.loc[df['category'] == 'baseline']
    records = []
    for index, row in df.iterrows():
        rec = dict()
        rec["id"] = row[0]
        rec["premise"] = row["premis"].strip(" .!?,").split(' ')
        rec["hypothesis"] = row["hypothesis"].strip(" .!?,").split(' ')
        rec["entailment"] = 1 if row["label"] == True else 0
        rec["type"] = row["category"]
        records.append(rec)
    return records



def accuracy(preds, ys):
    correct = 0.0
    for pred, y in zip(preds, ys):
        if pred == y:
            correct += 1.0
    return correct / float(len(ys))


def accuracyByCategory(preds, ys, types):
    possibleTypes = set(types)
    yToStrMap = {0: "non-entailment", 1: "entailment"}
    accDict = defaultdict(lambda: [0.0, 0.0])
    for pred, y, t in zip(preds, ys, types):
        label = yToStrMap[y]
        accDict["all"][0] += 1.0
        accDict[t][0] += 1.0
        accDict[label][0] += 1.0
        accDict[t + "_" + label][0] += 1.0
        if pred == y:
            accDict["all"][1] += 1.0
            accDict[t][1] += 1.0
            accDict[label][1] += 1.0
            accDict[t + "_" + label][1] += 1.0
    for k in accDict.keys():
        accDict[k] = accDict[k][1] / accDict[k][0]
    return dict(accDict)







class BiLMTextualEntailmentModel(nn.Module):
    def __init__(self, lstmSize = 30, hiddenSize = 300, h1 = 20):
        super().__init__()
        self.embedder = KeyedVectors.load_word2vec_format(datapath(WORD_2_VEC_PATH), binary = True)
        self.lm = nn.LSTM(hiddenSize + 2, lstmSize, bidirectional = True)
        self.l1 = nn.Linear(lstmSize * 2, h1)
        self.l2 = nn.Linear(h1, 1)

    def forward(self, premise, hypothesis):
        embedP = self.embed(premise)
        embedH = self.embed(hypothesis)
        embedSep = self.buildSepToken()
        embedC = torch.stack(embedP + [embedSep] + embedH).unsqueeze(1)
        _, (embedLSTM, _) = self.lm(embedC)
        embedLSTM = torch.flatten(embedLSTM)
        t = F.relu(self.l1(embedLSTM))
        y = torch.sigmoid(self.l2(t))
        return y

    def embed(self, sentence):
        embedList = []
        for w in sentence:
            try:
                wordEmbed = torch.from_numpy(self.embedder[w]).float()
                oobFlag = torch.FloatTensor([0])
                sepFlag = torch.FloatTensor([0])
                embedList.append(torch.cat((wordEmbed, oobFlag, sepFlag), 0))
            except KeyError:
                wordEmbed = torch.zeros((300,))
                oobFlag = torch.FloatTensor([1])
                sepFlag = torch.FloatTensor([0])
                embedList.append(torch.cat((wordEmbed, oobFlag, sepFlag), 0))
        return embedList

    def buildSepToken(self):
        wordEmbed = torch.zeros((300,))
        oobFlag = torch.FloatTensor([0])
        sepFlag = torch.FloatTensor([1])
        return torch.cat((wordEmbed, oobFlag, sepFlag), 0)





class TextualEntailmentClassifier:
    def __init__(self, model, lr = 0.00001):
        self.model = model
        self.lr = lr
        self.opt = optim.Adam(self.model.parameters(), lr = lr)
        self.bceLoss = nn.BCELoss()

    def train(self, trainDS, epochs = 30):
        losses = []
        for e in range(epochs):
            print("   epoch %d." % e)
            epochLoss = 0.0
            ys = []
            preds = []
            for i, rec in enumerate(trainDS):
                y = torch.FloatTensor([rec["entailment"]])
                pred = self.model(rec["premise"], rec["hypothesis"])
                preds.append(round(pred.item()))
                ys.append(rec["entailment"])
                #loss = ((pred - y) ** 2).mean()
                loss = self.bceLoss(pred, y)
                self.opt.zero_grad()
                loss.backward()
                epochLoss += loss.item()
                self.opt.step()
            losses.append(epochLoss)
            print("      Done. Loss = %f, Accuracy = %f." % (epochLoss, accuracy(preds, ys)), flush = True)
        return losses

    def run(self, runDS):
        results = []
        for i, rec in enumerate(runDS):
            pred = round(self.model(rec["premise"], rec["hypothesis"]).item())
            results.append(pred)
        return results

    def test(self, testDS):
        results = []
        epochLoss = 0.0
        for i, rec in enumerate(testDS):
            y = rec["entailment"]
            pred = round(self.model(rec["premise"], rec["hypothesis"]).item())
            results.append(pred)
            epochLoss += ((pred - y) ** 2)
        return (results, epochLoss)

    def freezeLM(self):
        for param in self.model.lm.parameters():
            param.requires_grad = False

    def unfreezeLM(self):
        for param in self.model.lm.parameters():
            param.requires_grad = True

    def resetHead(self, lr = 0.00001):
        torch.nn.init.xavier_uniform_(self.model.l1.weight)
        torch.nn.init.xavier_uniform_(self.model.l2.weight)
        self.opt = optim.Adam(self.model.parameters(), lr = lr)








def main():
    trainRecs = readData("./FinalData/train.csv", experimental = False)
    testRecs = readData("./FinalData/test.csv", experimental = False)
    model = BiLMTextualEntailmentModel()
    tec = TextualEntailmentClassifier(model)
    tec.resetHead()
    print("Training.", flush = True)
    tec.train(trainRecs)
    print("Testing.", flush = True)
    model.eval()
    with torch.no_grad():
        res, loss = tec.test(testRecs)
    #testAcc = accuracy(res, [rec["entailment"] for rec in testRecs])
    #print("   Baseline accuracy = %f." % testAcc, flush = True)
    testAcc = accuracyByCategory(res, [rec["entailment"] for rec in testRecs], [rec["type"] for rec in testRecs])
    print("   Baseline accuracy:\n%s" % testAcc, flush = True)
    trainRecs = readData("./FinalData/train.csv", experimental = True)
    testRecs = readData("./FinalData/test.csv", experimental = True)
    print("Freezing LM and resetting head.", flush = True)
    tec.freezeLM()
    tec.resetHead()
    print("Training.", flush = True)
    tec.train(trainRecs)
    print("Testing.", flush = True)
    model.eval()
    with torch.no_grad():
        res, loss = tec.test(testRecs)
    #testAcc = accuracy(res, [rec["entailment"] for rec in testRecs])
    #print("   Experimental accuracy (no retrain) = %f." % testAcc, flush = True)
    testAcc = accuracyByCategory(res, [rec["entailment"] for rec in testRecs], [rec["type"] for rec in testRecs])
    print("   Experimental accuracy (no retrain):\n%s" % testAcc, flush = True)
    trainRecs = readData("./FinalData/train.csv", experimental = True)
    testRecs = readData("./FinalData/test.csv", experimental = True)
    print("Unfreezing LM and resetting head.", flush = True)
    tec.unfreezeLM()
    tec.resetHead()
    print("Training.", flush = True)
    tec.train(trainRecs)
    print("Testing.", flush = True)
    model.eval()
    with torch.no_grad():
        res, loss = tec.test(testRecs)
    #testAcc = accuracy(res, [rec["entailment"] for rec in testRecs])
    #print("   Experimental accuracy (retrain) = %f." % testAcc, flush = True)
    testAcc = accuracyByCategory(res, [rec["entailment"] for rec in testRecs], [rec["type"] for rec in testRecs])
    print("   Experimental accuracy (retrain):\n%s" % testAcc, flush = True)




if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print("Done. Runtime = %f." % (end - start))

#===============================================================================
