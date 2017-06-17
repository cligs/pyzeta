#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: run_pyzeta.py
# author: #cf
# version: 0.2.0

import pyzeta
import os

"""
The pyzeta script is a Python implementation of Craig's Zeta.
Craig's Zeta is a measure of keyness or distinctiveness for contrastive analysis of two groups of texts.

See the readme.md and howto.md files for help on how to run the script.
"""


# =================================
# Zeta Parameters
# =================================

seglength = 2000  # int; 1000|2000|5000 are reasonable
pos = "all"  # Nc|Vv|Rg|Ag etc. depending on tagger model, or "all" if no selection
forms = "lemmata"  # words|lemmata|pos
contrast = ["detective", "yes", "no"]  # category, label1, label2
random = ["no", 20]


# =================================
# Files and folders
# =================================

workdir = "/media/christof/data/Dropbox/0-Analysen/2017/pyzeta/"  # full path to working directory; ends with slash
plaintextfolder = os.path.join(workdir, "sample-input", "corpus", "")
metadatafile = os.path.join(workdir, "sample-input", "metadata.csv")
stoplistfile = os.path.join(workdir, "sample-input", "stoplist.txt")
taggedfolder = os.path.join(workdir, "sample-output", "tagged", "")
datafolder = os.path.join(workdir, "sample-output", "data", "")
resultsfolder = os.path.join(workdir, "sample-output", "results", "")
contraststring = contrast[0] + "-" + contrast[1] + "-" + contrast[2]
parameterstring = str(seglength) + "-" + forms + "-" + str(pos)


# =============================================
# Function to run once for a given collection
# =============================================

# Prepare texts: tag and save (run once for a collection).
language = "en"  # TreeTagger language model code: fr|en|de|...
pyzeta.prepare(plaintextfolder, language, taggedfolder)


# =============================================
# Standard functions to calculate and plot zeta
# =============================================

# Calculate Zeta for words in two text collections
#pyzeta.zeta(taggedfolder, metadatafile, contrast, datafolder, resultsfolder, seglength, pos, forms, stoplistfile, random)


# Barchart with the most extreme zeta values
numfeatures = 50
#pyzeta.plot_zetascores(numfeatures, contrast, contraststring, parameterstring, resultsfolder)


# joint plot for random and real zeta scores
numfeatures = 100
#pyzeta.plot_realrandom(numfeatures, contrast, contraststring, parameterstring, resultsfolder)


# Scatterplot of types
numfeatures = 200  # int
cutoff = 0.40
#pyzeta.plot_types(numfeatures, cutoff, contrast, contraststring, parameterstring, resultsfolder)


# =============================================
# Extra functions (experimental)
# =============================================


# Threeway comparison (simple)
numfeatures = 25  # int
thirdgroup = ["subgenre", "tragicomedie"]  # category, label3
sortby = "comedy"  # label
mode = "generate"  # string; generate|analyze
# pyzeta.threeway_compare(datafolder, resultsfolder, contrast, contraststring, parameterstring, thirdgroup, numfeatures, sortby, mode)


# Threeway cluster analysis (dendrogram)
numfeatures = 25  # int
thirdgroup = ["subgenre", "tragicomedie"]  # category, label3
mode = "analyze" # string; generate|analyze
distmeasure = "euclidean"
#pyzeta.threeway_clustering(datafolder, resultsfolder, contrast, contraststring, parameterstring, thirdgroup, numfeatures, distmeasure, mode)

