from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
import pandas as pd
import csv

# SKLearn metric imports.
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Torch imports.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(13011)

def readData(filename):
    with open(filename) as csvfile:
        records = []
        dataReader = csv.reader(csvfile, delimiter = ',', quotechar = '\'')
        next(dataReader)
        for row in dataReader:
            rec = dict()
            rec["id"] = row[0]
            rec["premise"] = row[1].split(' ')
            rec["hypothesis"] = row[2].split(' ')
            rec["entailment"] = 1 if row[3] == "True" else 0
            rec["type"] = row[4]
            records.append(rec)
    return records

class TextualEntailmentClassifier:
    def __init__(self, lr = 0.0001):
        self.lr = lr
        self.lm = nn.LSTM(768, 200)
        self.l1 = nn.Linear(200, 80)
        #self.l1 = nn.Linear(2800,1)
        self.l2 = nn.Linear(80, 1)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.config = GPT2Config()

        # Load pre-trained model (weights)
        self.model = GPT2Model(self.config).from_pretrained('gpt2')

        self.opt = optim.Adam(self.model.parameters(), lr = lr)

        self.emb = self.model.get_input_embeddings()

    def embed(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)  # Batch size 1

        # only getting embedding for first word of sentence??
        return self.emb(input_ids)[0][0]

    def forward(self, premis, hypothesis):
        embedA = self.embed(premis)
        embedB = self.embed(hypothesis)

        embedC = torch.stack((embedA,embedB)).unsqueeze(1)

        _, (embedLSTM, _) = self.lm(embedC)

        embedLSTM = torch.flatten(embedLSTM)

        t = F.relu(self.l1(embedLSTM))

        y = torch.sigmoid(self.l2(t))

        return y


    def train(self, trainDS, epochs = 80):
        losses = []
        for e in range(epochs):
            print("   epoch %d." % e)
            epochLoss = 0.0
            for i, rec in enumerate(trainDS):
                y = rec["entailment"]
                pred = self.forward(rec["premise"], rec["hypothesis"])
                loss = ((pred - y) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                epochLoss += loss.item()
                self.opt.step()
            losses.append(epochLoss)
            print("      Done. Loss = %f." % epochLoss)
        return losses

    def test(self, testDS):
        results = []
        epochLoss = 0.0
        for i, rec in enumerate(testDS):
            y = rec["entailment"]
            pred = round(self.forward(rec["premise"], rec["hypothesis"]).item())
            results.append(pred)
            epochLoss += ((pred - y) ** 2)
        return (results, epochLoss)


def main():
    trainRecs = readData("./GeneratedDatasets/train.csv")
    validRecs = readData("./GeneratedDatasets/validate.csv")
    #testRecs = readData("./GeneratedDatasets/test.csv")

    #trainRecs = pd.DataFrame(trainRecs)
    #validRecs = pd.DataFrame(validRecs)

    tc = TextualEntailmentClassifier()
    print(tc.train(trainRecs,10))
    print(tc.test(validRecs))


if __name__ == '__main__':
    main()
