

import random
import sys
from string import Template
from timeit import default_timer as timer

random.seed(13011)



DEFAULT_DIRNAME = "./BuildingBlockDatasets/"
ROOT_KEY = ("ROOT",)
IN_TO_ON_LIST = ["the sun", "mercury", "venus", "earth", "mars", "saturn", "jupiter", "uranus", "neptune"]




def getArgs():
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
        if dirname[-1] != "/":
            dirname = dirname + "/"
    else:
        dirname = DEFAULT_DIRNAME
    return dirname




def getListDS(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    return lines




def buildTree(filename):
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
    treeDict[ROOT_KEY] = roots
    f.close()
    return treeDict




def buildCorrectPairsRec(tree, node, ancestors):
    pairs = []
    children = tree[node]
    if children == []:
        return []
    for child in children:
        for anc in ancestors:
            pairs.append((anc, child))
        pairs += buildCorrectPairsRec(tree, child, ancestors + [child])
    return pairs





def buildCorrectPairs(tree):
    roots = tree[ROOT_KEY]
    pairs = []
    for root in roots:
        pairs += buildCorrectPairsRec(tree, root, [root])
    return pairs




def buildIncorrectPairs(tree, goodPairs):
    badPairs = []
    for node1 in tree.keys():
        for node2 in tree.keys():
            if node1 != node2 and node1 != ROOT_KEY and node2 != ROOT_KEY and not (node1, node2) in goodPairs:
                badPairs.append((node1, node2))
    return badPairs





def buildRecords(correctPairs, extraDatasets):
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




def writeDataset(filename, ds):
    dsFile = open(filename, "w")
    for rec in ds:
        dsFile.write(str(rec))
        dsFile.write("\n")
    dsFile.close()




def main():
    dirname = getArgs()
    extraDatasets = dict()
    extraDatasets["animals"] = getListDS(dirname + "animals.txt")
    extraDatasets["jobs"] = getListDS(dirname + "jobs.txt")
    extraDatasets["movement"] = getListDS(dirname + "movement_verbs.txt")
    extraDatasets["activity"] = getListDS(dirname + "activities.txt")
    tree = buildTree(dirname + "compositions.csv")
    goodPairs = buildCorrectPairs(tree)
    badPairs = buildIncorrectPairs(tree, goodPairs)
    ent, nonEnt = buildRecords(goodPairs, extraDatasets)
    writeDataset("GeneratedDatasets/composition_entailment.txt", ent)
    writeDataset("GeneratedDatasets/composition_nonentailment.txt", nonEnt)





if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print("Done. Runtime = %f." % (end - start))

#===============================================================================
