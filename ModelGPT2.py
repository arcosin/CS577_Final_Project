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


class BiLMTextualEntailmentModel(nn.Module):
    def __init__(self, lstmSize = 200, hiddenSize = 1, h1 = 80):
        super().__init__()
        self.lm = nn.LSTM(hiddenSize + 2, lstmSize, bidirectional = True)
        self.l1 = nn.Linear(lstmSize * 2, h1)
        self.l2 = nn.Linear(h1, 1)

    def forward(self, premise, hypothesis):
        embedP = self.embed(premise)
        embedH = self.embed(hypothesis)
        embedSep = self.buildSepToken()
        embedC = torch.stack(embedP + [embedSep] + embedH).unsqueeze(1)
        embedLSTM, _ = self.lm(embedC)
        embedLSTM = embedLSTM.squeeze(1)
        t = F.relu(self.l1(embedLSTM))
        y = torch.sigmoid(self.l2(t))
        return y

    def embed(self, sentence):

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Encode a text inputs
        #text = "Who was Jim Henson ? Jim Henson was a"
        text = sentence
        indexed_tokens = tokenizer.encode(text)

        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])

        # Load pre-trained model (weights)
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        model.eval()

        # If you have a GPU, put everything on cuda
        #tokens_tensor = tokens_tensor.to('cuda')
        #model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
          
        oobFlag = torch.FloatTensor([0])
        sepFlag = torch.FloatTensor([0])

        result = []
        for i,w in enumerate(sentence):
            predicted_index = torch.argmax(predictions[0, -i, :]).item()
            item = torch.FloatTensor([predicted_index])
            result.append(torch.cat((item, oobFlag, sepFlag), 0))
     
        print(result)
        print(len(result))
        return result 

        # get the predicted next sub-word (in our case, the word 'man')
        #predicted_index = torch.argmax(predictions[0, -1, :]).item()
        #predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

    def buildSepToken(self):
        wordEmbed = torch.zeros((1,))
        oobFlag = torch.FloatTensor([0])
        sepFlag = torch.FloatTensor([1])
        return torch.cat((wordEmbed, oobFlag, sepFlag), 0)





class TextualEntailmentClassifier:
    def __init__(self, model, lr = 0.0001):
        self.model = model
        self.lr = lr
        self.opt = optim.Adam(self.model.parameters(), lr = lr)

    def train(self, trainDS, epochs = 120):
        losses = []
        for e in range(epochs):
            print("epoch %d." % e)
            epochLoss = 0.0
            for i, rec in enumerate(trainDS):
                preds = self.model(rec["premise"], rec["hypothesis"])
                loss = ((preds - ys) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                epochLoss += loss.item()
                self.opt.step()
            losses.append(epochLoss)
        return losses

    def run(self, runDS):
        results = []
        for i, rec in enumerate(runDS):
            pred = self.model(rec["premise"], rec["hypothesis"])
            results.append(pred)
        return results

    def freezeLM(self):
        for param in self.model.lm.parameters():
            param.requires_grad = False

    def unfreezeLM(self):
        for param in self.model.lm.parameters():
            param.requires_grad = True



def main():
    #trainRecs = readData("./GeneratedDatasets/train.csv")
    #validRecs = readData("./GeneratedDatasets/validate.csv")
    #testRecs = readData("./GeneratedDatasets/test.csv")
    model = BiLMTextualEntailmentModel()
    print(model.forward(["hello", "i", "am", "max"], ["hello", "i", "am", "here"]))



if __name__ == '__main__':
    main()


