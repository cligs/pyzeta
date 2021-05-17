#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: run_pyzeta.py
# author: #cf
# version: 0.3.0


"""
The pyzeta set of script is a Python implementation of Craig's Zeta and related measures.
Zeta is a measure of keyness or distinctiveness for contrastive analysis of two groups of texts.
This set of scripts does preprocessing, data preparation, score calculation, and visualization.
See the readme.md and howto.md files for help on how to run the script.
"""

# =================================
# Import statements
# =================================

import preprocess
import prepare
import calculate
import visualize
import experimental
from os.path import join


# =================================
# Parameters: files and folders
# =================================

# You need to adapt these
corpus = "corpus"
workdir = "D:/Downloads/roman20/Testset/"
dtmfolder = join(workdir, "output", "dtms", "")

# It is recommended to name your files and folders accordingly
#datadir = join(workdir, "data", corpus, "")
datafolder = join(workdir)
plaintextfolder = join(workdir, "corpus", "")
metadatafile = join(workdir, "metadata.csv")
stoplistfile = join(workdir, "stoplist.txt")

# It is recommended not to change these
outputdir = join(workdir, "output")
taggedfolder = join(outputdir, "tagged", "")
segmentfolder = join(outputdir, "segments1000", "")
#datafolder = join(outputdir, "results", "")
resultsfolder = join(outputdir, "results", "")
plotfolder = join(outputdir, "plots", "")


# =================================
# Preprocess
# =================================

"""
This module performs part-of-speech tagging on each text.
This module usually only needs to be called once when preparing a collection of texts.
Currently, this module uses TreeTagger and treetaggerwrapper.
"""

language = "fr"
sanitycheck = "no" # yes|no
#preprocess.main(plaintextfolder, taggedfolder, language, sanitycheck)


# =================================
# Prepare
# =================================

"""
This module performs several steps in preparing the data for analysis.
First, it splits each text into segments of a given length.
Second, it either takes all segments or samples an equal number of segments per text.
Third, it selects the desired features from each segment (form and pos)
Fourth, it creates document-term matrixes with absolute, relative and binary feature counts.
This function needs to be run again when a parameter is changed.
"""

segmentlength = 5000
max_num_segments = -1
featuretype = ["lemmata", "all"] # forms, pos
absolutefreqs, relativefreqs, binaryfreqs, absolutefreqs_sum, tf_frame = prepare.main(taggedfolder, segmentfolder,datafolder, dtmfolder, segmentlength, max_num_segments, stoplistfile, featuretype)


# =================================
# Calculate
# =================================

"""
This module performs the actual distinctiveness measure for each feature.
The calculation can be based on relative or binary features.
The calculation can work in several ways: by division, subtraction as well as with or without applying some log transformation.
The contrast parameter takes ["category", "group1", "group2"] as in the metadata table.
"""

separator = "\t"
#contrast = ["group", "early", "late"] # example for roman20
contrast = ["sentimental", "ja", "nein"] # for splitting groups randomly
logaddition= 0.1 # has effect on log calculation.
calculate.main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype, absolutefreqs, relativefreqs, binaryfreqs, absolutefreqs_sum, tf_frame)



# =================================
# Visualize
# =================================

"""
This module provides several plotting functionalities.
"zetabarchart" shows the n words with the most extreme, negative and postive, scores.
"typescatterplot" provides a scatterplot in which each dot is one feature.
"""

# This is for a horizontal barchart for plotting Zeta and similar scores per feature.
numfeatures = 25
measures = ["sd0", "sd2", "sg0", "sg2"]
#measures = ["sd0", "sd2", "sdX", "sr0", "sr2", "srX", "sg0", "dd0", "dd2", "ddX", "dr0", "dr2", "drX", "dg0"]
#droplist = ["anything", "everything", "anyone", "nothing"]
droplist = []
#visualize.zetabarchart(segmentlength, featuretype, contrast, measures, numfeatures, droplist, resultsfolder, plotfolder)

# This is for a scatterplot showing the relation between indicators and scores.
numfeatures = 500
measure = "sd0" # origzeta|logzeta|ratiorelfreqs|etc.
cutoff = 0.2
#visualize.typescatterplot(numfeatures, cutoff, contrast, segmentlength, featuretype, measure, resultsfolder, plotfolder)



# =================================
# Experimental
# =================================

"""
The function "comparisonplot" is a plot showing the top n features with the highest zeta scores for two measures in comparison.
"""

comparison = ["sd0", "dr0"]
# "sd0", "sd2", "sdX", "sr0", "sr2", "srX", "dd0", "dd2", "ddX", "dr0", "dr2", "drX"
numfeatures = 10
#experimental.comparisonplot(resultsfolder, plotfolder, comparison, numfeatures, segmentlength, featuretype, contrast)

"""
The function "get_correlation" calculates several correlation scores between the results of using different Zeta variants.
"""

comparison = ["sd0", "sd2", "sg0", "sg2", "dd0", "dd2", "dr0", "dr2"]
#experimental.get_correlation(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast)


numfeatures = 500
comparison = ["sd0", "sd2", "sg0", "sg2", "dd0", "dd2", "dr0", "dr2"]

#experimental.make_pca(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)
#experimental.make_dendrogram(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)
#experimental.make_tsne(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)

# TODO: The next step doesn't work in Spyder, it works in Jupyter... I don't understand why
#clustering_kmeans(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder, n=4)

#experimental.cluster_correlation(resultsfolder, segmentlength, featuretype, contrast, plotfolder, comparison)
