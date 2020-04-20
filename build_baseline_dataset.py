

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
        i += 1
        if i > maxRecordsPerTemplate:
            break
    premise = Template("The $J enjoys the $A1 and the $A2.")
    hypothesis1 = Template("The $J likes the $A1.")
    hypothesis2 = Template("The $J likes the $A2.")
    hypothesis3 = Template("The $J enjoys the $A2 and the $A1.")
    recTris = [(j, a1, a2) for j in ds["jobs"] for a1 in ds["animals"] for a2 in ds["animals"]]
    i = 0
    for job, a1, a2 in recTris:
        if a1 != a2:
            pCor = premise.substitute(J = job, A1 = a1, A2 = a2)
            hCor = hypothesis1.substitute(J = job, A1 = a1)
            correctRecords.append((pCor, hCor, True))
            hCor = hypothesis2.substitute(J = job, A2 = a2)
            correctRecords.append((pCor, hCor, True))
            hCor = hypothesis3.substitute(J = job, A1 = a1, A2 = a2)
            correctRecords.append((pCor, hCor, True))
            i += 1
            if i > maxRecordsPerTemplate:
                break
    premise = Template("$N is a $J with a pet $A and a house in $L.")
    hypothesis1 = Template("The $J has a pet $A.")
    hypothesis1 = Template("$N has a house in $L.")
    hypothesis3 = Template("The $A is a $J.")
    hypothesis4 = Template("$N has a pet $J.")
    recQuads = [(n, j, a, l) for n in ds["names"] for j in ds["jobs"] for a in ds["animals"] for l in ds["location"]]
    for n, j, a, l in recQuads:
        pCor = premise.substitute(N = n, J = j, A = a, L = l)
        hCor = hypothesis1.substitute(J = j, A = a)
        correctRecords.append((pCor, hCor, True))
        hCor = hypothesis2.substitute(N = n, L = l)
        correctRecords.append((pCor, hCor, True))
        hInc = hypothesis3.substitute(J = j, A = a)
        incorrectRecords.append((pCor, hInc, False))
        hInc = hypothesis3.substitute(N = n, J = j)
        incorrectRecords.append((pCor, hInc, False))
        i += 1
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
