

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
    inOrOn = lambda b: "on" if b else "in"
    premise = Template("The $A1 is $IN $L1.")
    hypothesis = Template("The $A1 is $IN $L2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        animal = random.choice(extraDatasets["animals"])
        pCor = premise.substitute(A1 = animal, L1 = w1, IN = inOrOn(w1 in IN_TO_ON_LIST))
        hCor = hypothesis.substitute(A1 = animal, L2 = w2, IN = inOrOn(w2 in IN_TO_ON_LIST))
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((hCor, pCor, False))
    premise = Template("The $J $V to $L1.")
    hypothesis = Template("The $J was in $L2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        job = random.choice(extraDatasets["jobs"])
        move = random.choice(extraDatasets["movement"])
        pCor = premise.substitute(J = job, V = move, L1 = w1, IN = inOrOn(w1 in IN_TO_ON_LIST))
        hCor = hypothesis.substitute(J = job, V = move, L2 = w2, IN = inOrOn(w2 in IN_TO_ON_LIST))
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((hCor, pCor, False))
    premise = Template("The $J is $A $IN $L1.")
    hypothesis = Template("The $J is $IN $L2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        job = random.choice(extraDatasets["jobs"])
        activity = random.choice(extraDatasets["activity"])
        pCor = premise.substitute(J = job, A = activity, L1 = w1, IN = inOrOn(w1 in IN_TO_ON_LIST))
        hCor = hypothesis.substitute(J = job, L2 = w2, IN = inOrOn(w2 in IN_TO_ON_LIST))
        pInc = premise.substitute(J = job, A = activity, L1 = w2, IN = inOrOn(w2 in IN_TO_ON_LIST))
        hInc = hypothesis.substitute(J = job, L2 = w1, IN = inOrOn(w1 in IN_TO_ON_LIST))
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((pInc, hInc, False))
    premise = Template("The $A1 is not $IN $L1.")
    hypothesis = Template("The $A1 could be $IN $L2.")
    for i, pair in enumerate(correctPairs):
        w2, w1 = pair
        animal = random.choice(extraDatasets["animals"])
        pCor = premise.substitute(A1 = animal, L1 = w1, IN = inOrOn(w2 in IN_TO_ON_LIST))
        hCor = hypothesis.substitute(A1 = animal, L2 = w2, IN = inOrOn(w1 in IN_TO_ON_LIST))
        pInc = premise.substitute(A1 = animal, L1 = w2, IN = inOrOn(w1 in IN_TO_ON_LIST))
        hInc = hypothesis.substitute(A1 = animal, L2 = w1, IN = inOrOn(w2 in IN_TO_ON_LIST))
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
    tree = buildTree(dirname + "locations.csv")
    goodPairs = buildCorrectPairs(tree)
    badPairs = buildIncorrectPairs(tree, goodPairs)
    ent, nonEnt = buildRecords(goodPairs, extraDatasets)
    writeDataset("GeneratedDatasets/geographic_entailment.txt", ent)
    writeDataset("GeneratedDatasets/geographic_nonentailment.txt", nonEnt)





if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print("Done. Runtime = %f." % (end - start))

#===============================================================================
