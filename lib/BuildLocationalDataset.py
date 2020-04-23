from . import BuildDataset
import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildLocationalDataset(BuildDataset.BuildDataset):
    def __init__(self, tree_infile, cor_outfile, inc_outfile, type):
        self.IN_TO_ON_LIST = ["the sun", "mercury", "venus", "earth", "mars", "saturn", "jupiter", "uranus", "neptune"]
        super().__init__(tree_infile, cor_outfile, inc_outfile, type)

    def buildRecords(self, correctPairs, extraDatasets):
        correctRecords = []
        incorrectRecords = []
        inOrOn = lambda b: "on" if b else "in"
        premise = Template("The $A1 is $IN $L1.")
        hypothesis = Template("The $A1 is $IN $L2.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair
            animal = random.choice(extraDatasets["animals"])
            pCor = premise.substitute(A1 = animal, L1 = w1, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            hCor = hypothesis.substitute(A1 = animal, L2 = w2, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((hCor, pCor, False))
        premise = Template("The $J $V to $L1.")
        hypothesis = Template("The $J was in $L2.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair
            job = random.choice(extraDatasets["jobs"])
            move = random.choice(extraDatasets["movement"])
            pCor = premise.substitute(J = job, V = move, L1 = w1, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            hCor = hypothesis.substitute(J = job, V = move, L2 = w2, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((hCor, pCor, False))
        premise = Template("The $J is $A $IN $L1.")
        hypothesis = Template("The $J is $IN $L2.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair
            job = random.choice(extraDatasets["jobs"])
            activity = random.choice(extraDatasets["activity"])
            pCor = premise.substitute(J = job, A = activity, L1 = w1, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            hCor = hypothesis.substitute(J = job, L2 = w2, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            pInc = premise.substitute(J = job, A = activity, L1 = w2, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            hInc = hypothesis.substitute(J = job, L2 = w1, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        premise = Template("The $A1 is not $IN $L1.")
        hypothesis = Template("The $A1 could be $IN $L2.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair
            animal = random.choice(extraDatasets["animals"])
            pCor = premise.substitute(A1 = animal, L1 = w1, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            hCor = hypothesis.substitute(A1 = animal, L2 = w2, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            pInc = premise.substitute(A1 = animal, L1 = w2, IN = inOrOn(w1 in self.IN_TO_ON_LIST))
            hInc = hypothesis.substitute(A1 = animal, L2 = w1, IN = inOrOn(w2 in self.IN_TO_ON_LIST))
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        return (correctRecords, incorrectRecords)

   
