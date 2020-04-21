from lib.BuildCompositionDataset import *
from lib.BuildLocationalDataset import *
from lib.BuildTemporalDataset import *
from lib.BuildHypernymDataset import *
from lib.BuildBaselineDataset import *
from lib.BuildHierarchicalTemporalDataset import *

BuildCompositionDataset("compositions.csv", "GeneratedDatasets/composition_entailment.txt","GeneratedDatasets/composition_nonentailment.txt","compositions")

BuildLocationalDataset("locations.csv", "GeneratedDatasets/locational_entailment.txt","GeneratedDatasets/locational_nonentailment.txt","compositions")

BuildTemporalDataset("compositions.csv", "GeneratedDatasets/temporal_entailment.txt","GeneratedDatasets/temporal_nonentailment.txt", "temporal")

BuildHypernymDataset("hypernyms_tree.csv", "GeneratedDatasets/hypernym_entailment.txt","GeneratedDatasets/hypernym_nonentailment.txt","compositions")

BuildHierarchicalTemporalDataset("temporal.csv", "GeneratedDatasets/htemporal_entailment.txt","GeneratedDatasets/htemporal_nonentailment.txt", "compositions")

#BuildBaselineDataset("compositions.csv", "GeneratedDatasets/baseline_entailment.txt","GeneratedDatasets/baseline_nonentailment.txt","compositions")
