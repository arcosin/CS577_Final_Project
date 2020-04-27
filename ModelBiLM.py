

import csv
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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



def readData(filename):
    with open(filename) as csvfile:
        dataReader = csv.reader(csvfile, delimiter = ',', quotechar = '\'')
        for row in dataReader:
            print(row)
            print(len(row))




class BiLMTextualEntailmentModel(nn.Module):
    def __init__(self, lstmSize = 200, hiddenSize = 300, h1 = 80):
        self.embedder = KeyedVectors.load_word2vec_format(datapath(WORD_2_VEC_PATH), binary = True)
        self.lm = nn.LSTM(hiddenSize, lstmSize, bidirectional = True)
        self.l1 = nn.Linear(lstmSize * 2, h1)
        self.l2 = nn.Linear(h1, 1)

    def forward(self, premise, hypothesis):
        embedP = self.embed(premise)
        embedH = self.embed(hypothesis)
        embedSep = buildSepToken()
        embedC = torch.stack(embedP + [embedSep] + embedH).unsqueeze(1)
        embedLSTM, _ = self.lstm(embedC).squeeze(1)
        t = F.relu(self.l1(embedLSTM))
        y = F.sigmoid(self.l2(t))
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







def main():
    readData("./GeneratedDatasets/test.csv")



if __name__ == '__main__':
    main()

#===============================================================================
