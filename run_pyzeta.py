#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: run_pyzeta.py
# author: #cf
# version: 0.1.0

import pyzeta


# =================================
# Zeta Parameters
# =================================

seglength = 2000
threshold = 10
mode = "tag"  # plain|tag|sel|posbigrams
pos = ["all"]  # Nc|Np|Vv|Rg|Ag etc., or "all" if no selection
forms = "lemmas"  # words|lemmas|pos
stoplist = ["De", "Et", "...", "qu'", "Qu'", "-là", "-ci", "C'est-à-dire", "c'est-à-dire", "Rome", "aux", "Aux", "l'"]
contrast = ["subgenre", "tragicomedie", "comedie"]  # Category, Label1, Label2


# =================================
# Files and folders
# =================================
workdir = "/media/christof/data/Dropbox/0-Analysen/2016/zeta/zeta2/"
plaintextfolder = workdir + "test/"
taggedfolder = workdir + "tagged/"
metadatafile = workdir + "metadata.csv"
datafolder = workdir + "data/"


# =================================
# Functions
# =================================

# Prepare texts: tag and save.
# pyzeta.prepare(plaintextfolder, taggedfolder)


# Calculate Zeta for words in two text collections
pyzeta.zeta(workdir, taggedfolder, metadatafile, contrast, datafolder, seglength, threshold, mode, pos, forms,
stoplist)


# Make a nice plot with some zeta data
zetafile = (datafolder + contrast[1] + "-" + contrast[2] + "_zeta-scores_segs-of-" +
            str(seglength) + "-" + mode + "-" + forms + "-" + str(pos[0]) + ".csv")
plotfile = (workdir + "zeta_scoreplot_" + contrast[1] + "-" + contrast[2] + "_segs-of-" +
            str(seglength) + "-" + mode + "-" + forms + "-" + str(pos[0]) + ".svg")
numwords = 25
# pyzeta.plot_zeta(zetafile, numwords, contrast, plotfile)


# Scatterplot of types
numfeatures = 1000
cutoff = 0.30
scatterfile = (workdir + "zeta_type-scatterplot_" + contrast[1] + "-" + contrast[2] +
               "_segs-of-" + str(seglength) + "-" + mode + "-" + forms + "-" + str(pos[0]) + ".svg")
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


# TODOS
# - Einmal TreeTagger laufen lassen, Ergebnis abspeichern, dann direkt darauf zugreifen
# - Vielleicht doch anders strukturieren: erst term-document-matrix mit absoluten Häufigkeiten für die Segmente, abspeichern. #
# - Dann binarisieren (vorhanden-nicht-vorhanden) und die Verhältnisse auszählen. Spart loops durch die Counter und Types.