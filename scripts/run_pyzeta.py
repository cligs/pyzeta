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
workdir = "/home/christof/repos/cligs/pyzeta/"
datadir = join(workdir, "data", "theatre")
corpus = "theatre"
dtmfolder = join("/home/christof/Desktop/pyzeta-dtms/", corpus, "")

# It is recommended to name your files and folders accordingly
plaintextfolder = join(datadir, "corpus", "")
metadatafile = join(datadir, "metadata.csv")
stoplistfile = join(datadir, "stoplist.txt")

# It is recommended not to change these
outputdir = join(workdir, "output", corpus)
taggedfolder = join(outputdir, "tagged", "")
segmentfolder = join(outputdir, "segments", "")
datafolder = join(outputdir, "results", "")
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

language = "en"
#preprocess.main(plaintextfolder, taggedfolder, language)


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

segmentlength = 1000
max_num_segments = -1
featuretype = ["lemmata", "Nc"] # forms, pos
#prepare.main(taggedfolder, segmentfolder, datafolder, dtmfolder, segmentlength, max_num_segments, stoplistfile, featuretype)


# =================================
# Calculate
# =================================

"""
This module performs the actual distinctiveness measure for each feature.
The calculation can be based on relative or binary features.
The calculation can work in several ways: by division, subtraction as well as with or without applying some log transformation.
"""

separator = "\t"
contrast = ["subgenre", "tragedie", "comedie"] # category, group1, group2
#contrast = ["random", "two", "one"]
logaddition= 0.1 # has effect on log calculation.
calculate.main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype)



# =================================
# Visualize
# =================================

"""
This module provides several plotting functionalities.
"zetabarchart" shows the n words with the most extreme, negative and postive, scores.
"typescatterplot" provides a scatterplot in which each dot is one feature.
"""

# This is for a horizontal barchart for plotting Zeta and similar scores per feature.
numfeatures = 20
measure = ["sd0", "sd2", "sdX", "sr0", "sr2", "srX", "dd0", "dd2", "ddX", "dr0", "dr2", "drX"] #droplist = ["anything", "everything", "anyone", "nothing"]
droplist = []
visualize.zetabarchart(segmentlength, featuretype, contrast, measure, numfeatures, droplist, resultsfolder, plotfolder)

# This is for a scatterplot showing the relation between indicators and scores.
numfeatures = 500
measure = "origzeta" # origzeta|logzeta|ratiorelfreqs|etc.
cutoff = 0.3
#visualize.typescatterplot(numfeatures, cutoff, contrast, segmentlength, featuretype, measure, resultsfolder, plotfolder)



# =================================
# Experimental
# =================================

"""
"comparisonplot" is a plot showing the top n features with the highest zeta scores for two measures in comparison.
"""

comparison = ["sd0", "dd0"]
# "sd0", "sd2", "sdX", "sr0", "sr2", "srX", "dd0", "dd2", "ddX", "dr0", "dr2", "drX"
numfeatures = 25
experimental.comparisonplot(resultsfolder, plotfolder, comparison, numfeatures, segmentlength, featuretype, contrast)

"""
"get_correlation" calculates several correlation scores between the results of using different Zeta variants.
"""

#experimental.get_correlation(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast)


"""
comparison = ['origzeta', 'log2zeta', 'log10zeta', 'divzeta', "ratiorelfreqs", "subrelfreqs", "logrelfreqs"]

for numfeatures in [10, 50, 100, 500, 1000, 2000]:

    make_pca(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)

    make_dendrogram(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)

    make_tsne(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder)

# TODO: The next step doesn't work in Spyder, it works in Jupyter... I don't understand why
#clustering_kmeans(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder, n=4)
"""
