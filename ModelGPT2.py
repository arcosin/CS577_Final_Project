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
        rec["premise"] = str(row["premis"]).strip(" .!?,").split(' ')
        rec["hypothesis"] = str(row["hypothesis"]).strip(" .!?,").split(' ')
        rec["entailment"] = 1 if row["label"] == True else 0
        rec["type"] = row["category"]
        records.append(rec)
    return records


def readData2(filename, category, label):
    df = pd.read_csv(filename)
    df = df.loc[df['category'] == category]
    df = df.loc[df['label'] == label]
    records = []
    for index, row in df.iterrows():
        rec = dict()
        rec["id"] = row[0]
        rec["premise"] = str(row["premis"]).strip(" .!?,").split(' ')
        rec["hypothesis"] = str(row["hypothesis"]).strip(" .!?,").split(' ')
        rec["entailment"] = 1 if row["label"] == True else 0
        rec["type"] = row["category"]
        records.append(rec)
    return records



class TextualEntailmentClassifier(nn.Module):
    def __init__(self, lr = 0.0001):
        super().__init__()
        self.lr = lr
        #self.lm = nn.Linear(768, 200)
        #self.l1 = nn.Linear(200*2, 80)
        self.l1 = nn.Linear(768, 80)
        self.l2 = nn.Linear(80*2, 1)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.config = GPT2Config()

        # Load pre-trained model (weights)
        self.model = GPT2Model(self.config).from_pretrained('gpt2')
        #self.model.cuda()

        self.opt = optim.Adam(self.model.parameters(), lr = lr)

        self.emb = self.model.get_input_embeddings()

    def embed(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0).cuda()  # Batch size 1

        embeds = self.emb(input_ids)
        #result = embeds[0][len(embeds[0])-1]

        # average each word vector to make sentence vector
        return torch.mean(self.emb(input_ids), dim=1)
        #return result

    def forward(self, premis, hypothesis):
        embedA = self.embed(premis)
        embedB = self.embed(hypothesis)
        embedC = torch.stack((embedA,embedB)).unsqueeze(1)
        #embedC = torch.cat((embedA,embedB),dim=1).unsqueeze(1)

        #_, (embedLSTM, _) = self.lm(embedC)
        embedLSTM = self.l1(embedC)

        embedLSTM = torch.flatten(embedLSTM)

        t = F.relu(embedLSTM)

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

def accuracy(preds, ys):
    try:
        correct = 0.0
        for pred, y in zip(preds, ys):
            if pred == y:
                correct += 1.0
        return correct / float(len(ys))
    except Exception as e:
        print(e)
        return 0

def main():
    torch.cuda.init()
    print("cuda",torch.cuda.device(0))
    print("available",torch.cuda.is_available())
    print("init",torch.cuda.is_initialized())

    IS_EXPERIMENTAL = True
    EPOCHS=80

    trainRecs = readData("./FinalData/train.csv", IS_EXPERIMENTAL)


    tc = TextualEntailmentClassifier(.0001)
    tc.cuda()
    tc.train(trainRecs,EPOCHS)


    valid_file = "./FinalData/validate.csv.bak"
    if IS_EXPERIMENTAL:
        validRecs = readData2(valid_file,'composition',False)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("composition", 0)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'composition',True)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("composition", 1)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'locational',False)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("locational", 0)
        print("   Accuracy = %f." % validAcc)


        validRecs = readData2(valid_file,'locational',True)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("locational", 1)
        print("   Accuracy = %f." % validAcc)


        validRecs = readData2(valid_file,'htemporal',False)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("htemporal", 0)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'htemporal',True)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("htemporal", 1)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'temporal',False)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("temporal", 0)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'temporal',True)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("temporal", 1)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'hypernym',False)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("hypernym", 0)
        print("   Accuracy = %f." % validAcc)

        validRecs = readData2(valid_file,'hypernym',True)
        res, loss = tc.test(validRecs)
        validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
        print("hypernym", 1)
        print("   Accuracy = %f." % validAcc)


    validRecs = readData(valid_file,IS_EXPERIMENTAL)
    res, loss = tc.test(validRecs)
    validAcc = accuracy(res, [rec["entailment"] for rec in validRecs])
    print("total")
    print("   Accuracy = %f." % validAcc)


if __name__ == '__main__':
    main()
