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

from os.path import join


# =================================
# Parameters: files and folders
# =================================

workdir = "/media/christof/data/repos/cligs/pyzeta/"
plaintextfolder = join(workdir, "sampledata", "corpus", "")
taggedfolder = join(workdir, "sampledata", "data", "tagged", "")
segmentfolder = join(workdir, "sampledata", "data", "segments", "") # should be parameter-dependent
metadatafile = join(workdir, "sampledata", "metadata.csv")
stoplistfile = join(workdir, "sampledata", "stoplist.txt")
datafolder = join(workdir, "sampledata", "data", "")
resultsfolder = join(workdir, "sampledata", "results", "")
plotfolder = join(workdir, "sampledata", "plots", "")


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
Second, it selects the desired features from each segment (form and pos)
Third, it creates document-term matrixes with absolute, relative and binary feature counts.
This function needs to be run again when a parameter is changed.
"""

segmentlength = 2000
featuretype = ["lemmata", "NN"] # forms, pos

#prepare.main(taggedfolder, segmentfolder, datafolder, segmentlength, stoplistfile, featuretype)


# =================================
# Calculate
# =================================

"""
This module performs the actual distinctiveness measure for each feature.
The calculation can be based on relative or binary features.
The calculation can work in several ways: by division, subtraction as well as with or without applying some log transformation.
"""

#contrast = ["title", "HoundBaskervilles", "LostWorld"] # category, group1, group2
contrast = ["subgenre", "detective", "historical"] # category, group1, group2

#calculate.main(datafolder, metadatafile, contrast, resultsfolder)




# =================================
# Visualize
# =================================

"""
This module provides several plotting functionalities.
"""

# This is for a horizontal barchart for plotting Zeta and similar scores per feature.
numfeatures = 20
measure = "origzeta"
visualize.zetabarchart(segmentlength, featuretype, contrast, measure, numfeatures, resultsfolder, plotfolder)

# This is for a scatterplot showing the relation between indicators and scores.
numfeatures = 2000
cutoff = 0.3
visualize.typescatterplot(numfeatures, cutoff, contrast, segmentlength, featuretype, measure, resultsfolder, plotfolder)
















