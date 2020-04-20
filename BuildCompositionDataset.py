import BuildDataset
import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildCompositionDataset(BuildDataset.BuildDataset):
    def __init__(self, tree_infile, cor_outfile, inc_outfile, type):
        super().__init__(tree_infile, cor_outfile, inc_outfile, type)

    def buildRecords(self, correctPairs, extraDatasets):
        correctRecords = []
        incorrectRecords = []

        # template 1
        premise = Template("$A1 are found in $A2.")
        hypothesis = Template("$A2 contain $A1.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair

            pCor = premise.substitute(A1 = w1, A2=w2)
            hCor = hypothesis.substitute(A1 = w1, A2=w2)

            pInc = premise.substitute(A1 = w1, A2=w2)
            hInc = hypothesis.substitute(A1 = w2, A2=w1)

            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))

        # template 2
        premise = Template("Some $J0 said $A1 consists of $A2.")
        hypothesis = Template("The $J0 said $A1 are composed of $A2.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hCor = hypothesis.substitute(J0 = job, A1 = w2, A2=w1)

                pInc = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hInc = hypothesis.substitute(J0 = job, A1 = w1, A2=w2)


                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        # template 3
        premise = Template("Some $A0 found $A1 in $A2.")
        hypothesis = Template("$A1 were found in $A2 by the $A0.")
        for animal in extraDatasets["animals"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(A1 = w1, A2=w2, A0 = animal)
                hCor = hypothesis.substitute(A1 = w1, A2=w2, A0 = animal)

                pInc = premise.substitute(A1 = w1, A2=w2, A0 = animal)
                hInc = hypothesis.substitute(A1 = w2, A2=w1, A0 = animal)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        # template 4
        premise = Template("Some $J0 said $A1 are made of $A2.")
        hypothesis = Template("The $J0 said $A2 are used to make $A1.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hCor = hypothesis.substitute(J0 = job, A1 = w2, A2=w1)

                pInc = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hInc = hypothesis.substitute(J0 = job, A1 = w1, A2=w2)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        #return (correctRecords, incorrectRecords)

        # template 5
        premise = Template("Some $J0 said wherever you find $A1 you'll find $A2.")
        hypothesis = Template("If you find $A1 you'll find $A2.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hCor = hypothesis.substitute(J0 = job, A1 = w2, A2=w1)

                pInc = premise.substitute(J0 = job, A1 = w2, A2=w1)
                hInc = hypothesis.substitute(J0 = job, A1 = w1, A2=w2)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        #return (correctRecords, incorrectRecords)


        # template 6
        premise = Template("Some $J0 came across some $A1.")
        hypothesis = Template("Some $J0 came across some $A1.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w2)
                hCor = hypothesis.substitute(J0 = job, A1 = w1)

                pInc = premise.substitute(J0 = job, A1 = w1)
                hInc = hypothesis.substitute(J0 = job, A1 = w2)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        #return (correctRecords, incorrectRecords)

        # template 7
        premise = Template("Some $J0 found some $A1.")
        hypothesis = Template("Some $J0 found some $A1.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w2)
                hCor = hypothesis.substitute(J0 = job, A1 = w1)

                pInc = premise.substitute(J0 = job, A1 = w1)
                hInc = hypothesis.substitute(J0 = job, A1 = w2)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        #return (correctRecords, incorrectRecords)

        # template 8
        premise = Template("Some $J0 came across some $A1 in some $A2.")
        hypothesis = Template("Some $J0 came across some $A1 in some $A2.")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

                pCor = premise.substitute(J0 = job, A1 = w1, A2=w2)
                hCor = hypothesis.substitute(J0 = job, A1 = w1, A2=w2)

                pInc = premise.substitute(J0 = job, A1 = w1, A2=w2)
                hInc = hypothesis.substitute(J0 = job, A1 = w2, A2=w1)

                correctRecords.append((pCor, hCor, True))
                incorrectRecords.append((pInc, hInc, False))

        #return (correctRecords, incorrectRecords)

        # template 9
        premise = Template("$A1 are not found in $A2.")
        hypothesis = Template("$A2 do not contain $A1.")
        for i, pair in enumerate(correctPairs):
            w2, w1 = pair

            pCor = premise.substitute(A1 = w1, A2=w2)
            hCor = hypothesis.substitute(A1 = w1, A2=w2)

            pInc = premise.substitute(A1 = w1, A2=w2)
            hInc = hypothesis.substitute(A1 = w2, A2=w1)

            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))

        return (correctRecords, incorrectRecords)
        


BuildCompositionDataset("compositions.csv", "GeneratedDatasets/composition_entailment.txt","GeneratedDatasets/composition_nonentailment.txt","compositions")
