from lib.BuildCompositionDataset import *
from lib.BuildLocationalDataset import *
from lib.BuildTemporalDataset import *
from lib.BuildHypernymDataset import *
from lib.BuildBaselineDataset import *
from lib.BuildHierarchicalTemporalDataset import *
import os
import random
import pandas as pd

def buildAll():
    print("Building compositions")
    BuildCompositionDataset("compositions.csv", "GeneratedDatasets/composition_entailment.txt","GeneratedDatasets/composition_nonentailment.txt","compositions")

    print("Building locational")
    BuildLocationalDataset("locations.csv", "GeneratedDatasets/locational_entailment.txt","GeneratedDatasets/locational_nonentailment.txt","compositions")

    print("Building temporal")
    BuildTemporalDataset("compositions.csv", "GeneratedDatasets/temporal_entailment.txt","GeneratedDatasets/temporal_nonentailment.txt", "temporal")

    print("Building hypernym")
    BuildHypernymDataset("hypernyms_tree.csv", "GeneratedDatasets/hypernym_entailment.txt","GeneratedDatasets/hypernym_nonentailment.txt","compositions")

    print("Building htemporal")
    BuildHierarchicalTemporalDataset("temporal.csv", "GeneratedDatasets/htemporal_entailment.txt","GeneratedDatasets/htemporal_nonentailment.txt", "compositions")

    print("Building baseline")
    BuildBaselineDataset("compositions.csv", "GeneratedDatasets/baseline_entailment.txt","GeneratedDatasets/baseline_nonentailment.txt","compositions")

def finalizeData(MAX_EXAMPLES = 100, SPLIT_PERC = 10):
    train_df = pd.DataFrame(columns=['hypothesis','premis','label', 'category'])
    val_df = pd.DataFrame(columns=['hypothesis','premis','label', 'category'])
    test_df = pd.DataFrame(columns=['hypothesis','premis','label', 'category'])

    # look through each file in generated datasets
    for subdir, dirs, files in os.walk("GeneratedDatasets"):
        for file_name in files:
            # get the full file path
            filepath = subdir + os.sep + file_name

            # display which file were on to track progress
            print(filepath)

            # open the file
            f = open(filepath, 'r')

            # get number of lines in file
            lines = f.readlines()

            # if the number of lines is > MAX_EXAMPLES 
            # then randomly choose MAX_EXAMPLE lines to keep in lines_to_append
            while len(lines) > MAX_EXAMPLES:
                del lines[random.randint(0,len(lines)-1)]

            # category is the first part of the file name 
            category = file_name.split('_')[0]

            # loop through the lines to include 
            for line in lines:
                # strip special characters
                line = line.strip('()\n\'\"')

                # split hyp/prem/label
                line = line.split(',')

                # form our new row to append to result
                new_row = {'hypothesis':line[0], 'premis':line[1], 'label':line[2].strip(), 'category':category}

                # append the new row to result
                which_set = random.randint(0,100)

                if which_set < SPLIT_PERC:
                    test_df = test_df.append(new_row, ignore_index=True)
                elif which_set >= SPLIT_PERC and which_set < SPLIT_PERC*2:
                    val_df = val_df.append(new_row, ignore_index=True)
                else:
                    train_df = train_df.append(new_row, ignore_index=True)


    train_df.to_csv("GeneratedDatasets/train.csv")
    test_df.to_csv("GeneratedDatasets/test.csv")
    val_df.to_csv("GeneratedDatasets/validate.csv")



buildAll()
finalizeData(10000,10)
