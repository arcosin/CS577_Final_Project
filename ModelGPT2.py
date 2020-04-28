from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Load pre-trained model (weights)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.opt = optim.Adam(self.model.parameters(), lr = lr)

    def embed(self, sentence):
        # Encode a text inputs
        #text = "Who was Jim Henson ? Jim Henson was a"
        text = sentence
        indexed_tokens = self.tokenizer.encode(text)

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.model.eval()

        # If you have a GPU, put everything on cuda
        #tokens_tensor = tokens_tensor.to('cuda')
        #model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
          
        #predicted_index = torch.argmax(predictions[0, -1, :]).item()
        #predicted_text = self.tokenizer.decode(indexed_tokens + [predicted_index])
        #return torch.FloatTensor(predictions)
        return predictions
        #return predicted_text


    def forward(self, premis, hypothesis):
        embedA = self.embed(premis)
        embedB = self.embed(hypothesis)

        embedC = torch.stack((embedA, embedB)).unsqueeze(1)

        embedLSTM = nn.LSTM(embedC)

        embedLSTM = torch.flatten(embedLSTM)
        t = F.relu(nn.Linear(embedLSTM))
        y = torch.sigmoid(nn.Linear(t))
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
