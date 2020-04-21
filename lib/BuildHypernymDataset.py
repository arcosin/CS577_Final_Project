from . import BuildDataset
import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildHypernymDataset(BuildDataset.BuildDataset):
    def __init__(self, tree_infile, cor_outfile, inc_outfile, type):
        super().__init__(tree_infile, cor_outfile, inc_outfile, type)

    def buildRecords(self, correctPairs, extraDatasets):
        correctRecords = []
        incorrectRecords = []
        premise = Template("There is not a $W in $L.")
        hypothesis = Template("There is not a $W in $L.")
        for i, pair in enumerate(correctPairs):
            hyper, hypo = pair
            loc = random.choice(extraDatasets["locations"])
            pCor = premise.substitute(W = hyper, L = loc)
            hCor = hypothesis.substitute(W = hypo, L = loc)
            loc = random.choice(extraDatasets["locations"])
            pInc = premise.substitute(W = hypo, L = loc)
            hInc = hypothesis.substitute(W = hyper, L = loc)
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        premise = Template("The $J only has a $W1 and a $W2.")
        hypothesis = Template("The $J has two ${W1}s.")
        for i, pair in enumerate(correctPairs):
            hyper, hypo = pair
            job = random.choice(extraDatasets["jobs"])
            pCor = premise.substitute(J = job, W1 = hyper, W2 = hypo)
            hCor = hypothesis.substitute(J = job, W1 = hyper)
            pInc = premise.substitute(J = job, W1 = hyper, W2 = hypo)
            hInc = hypothesis.substitute(J = job, W1 = hypo)
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        premise = Template("There is a $W in $L.")
        hypothesis = Template("There is a $W in $L.")
        for i, pair in enumerate(correctPairs):
            hyper, hypo = pair
            loc = random.choice(extraDatasets["locations"])
            pCor = premise.substitute(W = hypo, L = loc)
            hCor = hypothesis.substitute(W = hyper, L = loc)
            loc = random.choice(extraDatasets["locations"])
            pInc = premise.substitute(W = hyper, L = loc)
            hInc = hypothesis.substitute(W = hypo, L = loc)
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        premise = Template("The $J purchased a $W.")
        hypothesis = Template("The $J bought a $W.")
        for i, pair in enumerate(correctPairs):
            hyper, hypo = pair
            job = random.choice(extraDatasets["jobs"])
            pCor = premise.substitute(J = job, W = hypo)
            hCor = hypothesis.substitute(J = job, W = hyper)
            pInc = premise.substitute(J = job, W = hyper)
            hInc = hypothesis.substitute(J = job, W = hypo)
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        premise = Template("The $J is living with a $W.")
        hypothesis = Template("The $J has a $W in his home.")
        for i, pair in enumerate(correctPairs):
            hyper, hypo = pair
            job = random.choice(extraDatasets["jobs"])
            pCor = premise.substitute(J = job, W = hypo)
            hCor = hypothesis.substitute(J = job, W = hyper)
            pInc = premise.substitute(J = job, W = hyper)
            hInc = hypothesis.substitute(J = job, W = hypo)
            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))
        return (correctRecords, incorrectRecords)

