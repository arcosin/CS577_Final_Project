from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
          
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_text = self.tokenizer.decode(indexed_tokens + [predicted_index])
        return predicted_text

    def forward(self, premis, hypothesis):
        return self.embed(premis) == hypothesis


    def train(self, trainDS, epochs = 120):
        losses = []
        for e in range(epochs):
            print("epoch %d." % e)
            epochLoss = 0.0
            for i, rec in enumerate(trainDS):
                preds = self.forward(rec["premise"], rec["hypothesis"])
                loss = ((preds - rec["entailment"]) ** 2)
                self.opt.zero_grad()
                #loss.backward()
                #epochLoss += loss.item()
                epochLoss += loss
                self.opt.step()
            losses.append(epochLoss)
        return losses

    def run(self, runDS):
        results = []
        for i, rec in enumerate(runDS):
            pred = self.forward(rec["premise"], rec["hypothesis"])
            results.append(pred)
        return results



def main():
    trainRecs = readData("./GeneratedDatasets/train.csv")
    validRecs = readData("./GeneratedDatasets/validate.csv")
    #testRecs = readData("./GeneratedDatasets/test.csv")

    tc = TextualEntailmentClassifier()
    print(tc.train(trainRecs,10))
    print(tc.run(validRecs))


if __name__ == '__main__':
    main()


