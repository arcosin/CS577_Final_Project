from lib.BuildCompositionDataset import *
from lib.BuildLocationalDataset import *
from lib.BuildTemporalDataset import *
from lib.BuildHypernymDataset import *

BuildCompositionDataset("compositions.csv", "GeneratedDatasets/composition_entailment.txt","GeneratedDatasets/composition_nonentailment.txt","compositions")

BuildLocationalDataset("locations.csv", "GeneratedDatasets/locational_entailment.txt","GeneratedDatasets/locational_nonentailment.txt","compositions")

BuildTemporalDataset("compositions.csv", "GeneratedDatasets/temporal_entailment.txt","GeneratedDatasets/temporal_nonentailment.txt", "temporal")

BuildHypernymDataset("hypernyms_tree.csv", "GeneratedDatasets/hypernym_entailment.txt","GeneratedDatasets/hypernym_nonentailment.txt","compositions")
