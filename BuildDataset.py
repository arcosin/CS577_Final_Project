import random
import sys
from string import Template
from timeit import default_timer as timer

class BuildDataset():
    def __init__(self, tree_infile, inc_outfile, cor_outfile):
        random.seed(13011)
        self.DEFAULT_DIRNAME = "./BuildingBlockDatasets/"
        self.ROOT_KEY = ("ROOT",)

        self.dirname = self.getArgs()
        self.extraDatasets = dict()
        self.extraDatasets["animals"] = self.getListDS(self.dirname + "animals.txt")
        self.extraDatasets["jobs"] = self.getListDS(self.dirname + "jobs.txt")
        self.extraDatasets["movement"] = self.getListDS(self.dirname + "movement_verbs.txt")
        self.extraDatasets["activity"] = self.getListDS(self.dirname + "activities.txt")
        self.tree = self.buildTree(self.dirname + tree_infile)
        self.goodPairs = self.buildCorrectPairs(self.tree)
        self.badPairs = self.buildIncorrectPairs(self.tree, self.goodPairs)
        self.ent, self.nonEnt = self.buildRecords(self.goodPairs, self.extraDatasets)
        self.writeDataset("GeneratedDatasets/composition_entailment.txt", self.ent)
        self.writeDataset("GeneratedDatasets/composition_nonentailment.txt", self.nonEnt)

    def getArgs(self):
        if len(sys.argv) > 1:
            dirname = sys.argv[1]
            if dirname[-1] != "/":
                dirname = dirname + "/"
        else:
            dirname = self.DEFAULT_DIRNAME
        return dirname

    def getListDS(self,filename):
        f = open(filename, 'r')
        lines = f.read().splitlines()
        f.close()
        return lines

    def buildTree(self,filename):
        f = open(filename, 'r')
        treeDict = dict()
        lines = f.read().splitlines()
        setOfNonRoots = set()
        for line in lines:
            line = line.split(",")
            node = line[0]
            children = line[1:]
            setOfNonRoots.update(children)
            treeDict[node] = children
        edgeLists = list(treeDict.values())
        nodes = list(treeDict.keys())
        for edgeList in edgeLists:
            for valNode in edgeList:
                if valNode not in nodes:
                    treeDict[valNode] = []
        roots = list(set(treeDict.keys()) - setOfNonRoots)
        treeDict[self.ROOT_KEY] = roots
        f.close()
        return treeDict

    def buildCorrectPairsRec(self, tree, node, ancestors):
        pairs = []
        children = tree[node]
        if children == []:
            return []
        for child in children:
            for anc in ancestors:
                pairs.append((anc, child))
            pairs += self.buildCorrectPairsRec(tree, child, ancestors + [child])
        return pairs

    def buildCorrectPairs(self, tree):
        roots = tree[self.ROOT_KEY]
        pairs = []
        for root in roots:
            pairs += self.buildCorrectPairsRec(tree, root, [root])
        return pairs

    def buildIncorrectPairs(self, tree, goodPairs):
        badPairs = []
        for node1 in tree.keys():
            for node2 in tree.keys():
                if node1 != node2 and node1 != self.ROOT_KEY and node2 != self.ROOT_KEY and not (node1, node2) in goodPairs:
                    badPairs.append((node1, node2))
        return badPairs

    def buildRecords(self, correctPairs, extraDatasets):
        print("buildRecords Not implemented")
        exit()

        correctRecords = []
        incorrectRecords = []

        # template 1
        premise = Template("$A1 are found in $A2.")
        hypothesis = Template("$A2 contain $A1")
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
        hypothesis = Template("The $J0 said $A1 are composed of $A2")
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
        hypothesis = Template("$A1 were found in $A2 by the $A0")
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
        hypothesis = Template("The $J0 said $A2 are used to make $A1")
        for job in extraDatasets["jobs"]:
            for i, pair in enumerate(correctPairs):
                w2, w1 = pair

            pCor = premise.substitute(J0 = job, A1 = w2, A2=w1)
            hCor = hypothesis.substitute(J0 = job, A1 = w2, A2=w1)

            pInc = premise.substitute(J0 = job, A1 = w2, A2=w1)
            hInc = hypothesis.substitute(J0 = job, A1 = w1, A2=w2)

            correctRecords.append((pCor, hCor, True))
            incorrectRecords.append((pInc, hInc, False))

        return (correctRecords, incorrectRecords)

    def writeDataset(self, filename, ds):
        dsFile = open(filename, "w")
        for rec in ds:
            dsFile.write(str(rec))
            dsFile.write("\n")
        dsFile.close()




