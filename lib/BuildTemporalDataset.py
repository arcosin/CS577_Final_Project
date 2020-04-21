from . import BuildDataset
import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildTemporalDataset(BuildDataset.BuildDataset):
    def __init__(self, tree_infile, cor_outfile, inc_outfile, type):
        super().__init__(tree_infile, cor_outfile, inc_outfile, type)

    def buildRecords(self, extraDatasets):
        correctRecords = []
        incorrectRecords = []

        # template 1
        premise = Template("The $J0 went to $L0 $T0.")
        hypothesis = Template("The $J0 visited $L0")

        for temp in extraDatasets["tempModifiers"]:
            job = random.choice(extraDatasets["jobs"])
            location = random.choice(extraDatasets["locations"])

            pCor = premise.substitute(J0 = job, L0 = location, T0 = temp)
            hCor = hypothesis.substitute(J0 = job, L0 = location)

            correctRecords.append((pCor, hCor, True))


        # template 2
        premise = Template("The $J1 went to $L1.")
        hypothesis = Template("The $J1 visited $L1 $T1")

        for temp in extraDatasets["tempModifiers"]:
            job = random.choice(extraDatasets["jobs"])
            location = random.choice(extraDatasets["locations"])

            pInc = premise.substitute(J1 = job, L1 = location)
            hInc = hypothesis.substitute(J1 = job, L1 = location, T1 = temp)

            incorrectRecords.append((pInc, hInc, False))


        return (correctRecords, incorrectRecords)

        # template 3
        premise = Template("The $J1 remembers going to $L1 $T1.")
        hypothesis = Template("The $J1 visited $L1")

        for temp in extraDatasets["tempModifiers"]:
            job = random.choice(extraDatasets["jobs"])
            location = random.choice(extraDatasets["locations"])

            pInc = premise.substitute(J1 = job, L1 = location, T1 = temp)
            hInc = hypothesis.substitute(J1 = job, L1 = location)

            incorrectRecords.append((pInc, hInc, False))


        return (correctRecords, incorrectRecords)


