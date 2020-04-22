from . import BuildDataset
import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildHierarchicalTemporalDataset(BuildDataset.BuildDataset):
    def __init__(self, tree_infile, cor_outfile, inc_outfile, type):
        super().__init__(tree_infile, cor_outfile, inc_outfile, type)

    def buildRecords(self, correctPairs, extraDatasets):
        correctRecords = []
        incorrectRecords = []

        # template 1
        premise = Template("The $J1 went to $L1 more than a $T1 ago")
        hypothesis = Template("The $J1 went to $L1 more than a $T2 ago.")
        for i, pair in enumerate(correctPairs):
            for job in extraDatasets["jobs"]:
                location = random.choice(extraDatasets["locations"])
                w2, w1 = pair

                pCor = premise.substitute(J1= job, L1 = location, T1=w2)
                hCor = hypothesis.substitute(J1=job, L1 = location, T2=w1)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((hCor, pCor, False))

        return (correctRecords, incorrectRecords)

