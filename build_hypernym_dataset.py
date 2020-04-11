

import random
import sys
from string import Template
from timeit import default_timer as timer

random.seed(13011)



DEFAULT_DIRNAME = "./BuildingBlockDatasets/"
ROOT_KEY = ("ROOT",)




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
    premise = Template("A $W1 is a $W2.")
    hypothesis = Template("A $W1 is a $W2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        pCor = premise.substitute(W1 = w1, W2 = w2)
        hCor = hypothesis.substitute(W1 = w1, W2 = w2)
        pInc = premise.substitute(W1 = w1, W2 = w2)
        hInc = hypothesis.substitute(W1 = w2, W2 = w1)
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((pInc, hInc, False))
    premise = Template("A $W1 is a $W2.")
    hypothesis = Template("A $W1 is not necessarily a $W2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        pCor = premise.substitute(W1 = w1, W2 = w2)
        hCor = hypothesis.substitute(W1 = w2, W2 = w1)
        pInc = premise.substitute(W1 = w1, W2 = w2)
        hInc = hypothesis.substitute(W1 = w1, W2 = w2)
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((pInc, hInc, False))
    premise = Template("There is a $W in $L.")
    hypothesis = Template("There is a $W in $L.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        loc = random.choice(extraDatasets["location"])
        pCor = premise.substitute(W = w1, L = loc)
        hCor = hypothesis.substitute(W = w2, L = loc)
        loc = random.choice(extraDatasets["location"])
        pInc = premise.substitute(W = w2, L = loc)
        hInc = hypothesis.substitute(W = w1, L = loc)
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
    extraDatasets["location"] = getListDS(dirname + "locations.txt")
    tree = buildTree(dirname + "hypernyms_tree.csv")
    goodPairs = buildCorrectPairs(tree)
    ent, nonEnt = buildRecords(goodPairs, extraDatasets)
    writeDataset("GeneratedDatasets/hypernym_entailment.txt", ent)
    writeDataset("GeneratedDatasets/hypernym_nonentailment.txt", nonEnt)





if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print("Done. Runtime = %f." % (end - start))

#===============================================================================
