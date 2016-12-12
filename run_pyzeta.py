#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: run_pyzeta.py
# author: #cf
# version: 0.2.0

import pyzeta


# =================================
# Zeta Parameters
# =================================

seglength = 1000
pos = "all"  # Nc|Np|Vv|Rg|Ag etc., or "all" if no selection
forms = "lemmata"  # words|lemmata|pos
contrast = ["subgenre", "tragedie", "comedie"]  # Category, Label1, Label2


# =================================
# Files and folders
# =================================
workdir = "/media/christof/data/Dropbox/0-Analysen/2016/zeta/zeta2/"
plaintextfolder = workdir + "text/"
taggedfolder = workdir + "tagged/"
metadatafile = workdir + "metadata.csv"
datafolder = workdir + "data/"
resultsfolder =  workdir + "results/"
stoplistfile = workdir + "stoplist.txt"


# =================================
# Functions
# =================================

# Prepare texts: tag and save (run once for a collection).
# pyzeta.prepare(plaintextfolder, taggedfolder)


# Calculate Zeta for words in two text collections
#pyzeta.zeta(taggedfolder, metadatafile, contrast, datafolder, resultsfolder,
#            seglength, pos, forms, stoplistfile)


# Make a nice plot with some zeta data
numwords = 25
pyzeta.plot_zeta(numwords, contrast, seglength, pos, forms, resultsfolder)


# Scatterplot of types
numfeatures = 1000
cutoff = 0.30
scatterfile = (workdir + "zeta_type-scatterplot_" + contrast[1] + "-" + contrast[2] +
               "_segs-of-" + str(seglength) + "-" + forms + "-" + str(pos[0]) + ".svg")
# pyzeta.plot_types(zetafile, numfeatures, cutoff, contrast, scatterfile)


# Threeway comparison
numfeatures = 20
components = [1, 2]
threecontrast = [["subgenre", "comedie", "tragedie", "tragicomedie"],
                 ["comedie", "comedie", "other"],
                 ["tragedie", "tragedie", "other"],
                 ["tragicomedie", "tragicomedie", "other"]]
# pyzeta.threeway(datafolder, zetafile, numfeatures, components, plaintextfolder, metadatafile,
#                threecontrast, seglength, mode, pos, forms, stoplist)