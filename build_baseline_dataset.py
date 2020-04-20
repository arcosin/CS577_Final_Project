

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





def buildRecords(ds, maxRecordsPerTemplate = 2000):
    correctRecords = []
    incorrectRecords = []
    premise = Template("There is a $A in $L.")
    hypothesis = Template("A $A can be found in $L.")
    recPairs = [(a, l) for a in ds["animals"] for l in ds["location"]]
    i = 0
    for animal, location in recPairs:
        pCor = premise.substitute(A = animal, L = location)
        hCor = hypothesis.substitute(A = animal, L = location)
        wrongLoc = random.choice(ds["location"])
        while wrongLoc == location:
            wrongLoc = random.choice(ds["location"])
        pInc = premise.substitute(A = animal, L = location)
        hInc = hypothesis.substitute(A = animal, L = wrongLoc)
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((pInc, hInc, False))
        i += 2
        if i > maxRecordsPerTemplate:
            break
    premise = Template("There was a $A in $L.")
    hypothesis = Template("A $A could be found in $L.")
    recPairs = [(a, l) for a in ds["animals"] for l in ds["location"]]
    i = 0
    for animal, location in recPairs:
        pCor = premise.substitute(A = animal, L = location)
        hCor = hypothesis.substitute(A = animal, L = location)
        wrongAni = random.choice(ds["animals"])
        while wrongAni == animal:
            wrongAni = random.choice(ds["animals"])
        pInc = premise.substitute(A = animal, L = location)
        hInc = hypothesis.substitute(A = wrongAni, L = location)
        correctRecords.append((pCor, hCor, True))
        incorrectRecords.append((pInc, hInc, False))
        i += 2
        if i > maxRecordsPerTemplate:
            break
    premise = Template("$N doesn't like $A in $L.")
    hypothesis1 = Template("$N doesn't like $A.")
    hypothesis2 = Template("$N doesn't like $L.")
    hypothesis3 = Template("$N likes $A in $L.")
    recTris = [(n, a, l) for n in ds["names"] for a in ds["activity"] for l in ds["location"]]
    i = 0
    for name, activity, location in recTris:
        pInc = premise.substitute(N = name, A = activity, L = location)
        hInc = hypothesis1.substitute(N = name, A = activity)
        incorrectRecords.append((pInc, hInc, False))
        hInc = hypothesis2.substitute(N = name, L = location)
        incorrectRecords.append((pInc, hInc, False))
        hInc = hypothesis3.substitute(N = name, A = activity, L = location)
        incorrectRecords.append((pInc, hInc, False))
        i += 3
        if i > maxRecordsPerTemplate:
            break
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
    extraDatasets["location"] = getListDS(dirname + "locations.txt")
    extraDatasets["names"] = getListDS(dirname + "names.txt")
    ent, nonEnt = buildRecords(extraDatasets)
    print("%d entailment examples made." % len(ent))
    print("%d non-entailment examples made." % len(nonEnt))
    writeDataset("GeneratedDatasets/baseline_entailment.txt", ent)
    writeDataset("GeneratedDatasets/baseline_nonentailment.txt", nonEnt)





if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print("Done. Runtime = %f." % (end - start))

#===============================================================================
