#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: run_pyzeta.py
# author: #cf
# version: 0.2.0

import pyzeta

"""
The pyzeta script is a Python implementation of Craig's Zeta.
Craig's Zeta is a measure of keyness or distinctiveness for contrastive analysis of two groups of texts.

Currently, the following processes are supported:
- Prepare a text collection by tagging it using TreeTagger (pyzeta.prepare; run once per collection)
- For any two partitions of the collection, create a matrix of per-segment type frequencies
  and calculate the zeta scores for the vocabulary (pyzeta.zeta).
  There are options to choose word forms or lemmata or POS as features.
  There is the possibility to filter features based on their POS.
- Visualize the most distinctive words as a horizontal bar chart. (pyzeta.plot_scores)
- Visualize the feature distribution as a scatterplot (pyzeta.plot_types)
- Currently non-functional: PCA for three partitions using distinctive features.

The script expects the following as input:
- A folder with plain text files
- A metadata file with category information about each file, identified through "idno" = filename
- A file with stopwords, one per line

You can set the following parameters:
- How many word forms should each segment have (500-5000 may be reasonable)
- Which POS should be selected? "all" selects all, "Xy" will select words corresponding to Xy POS tag.
- Which partitions of the data should be contrasted? Indicate the category and the two contrasting labels.

For more information on Zeta, see:
- Burrows, John. „All the Way Through: Testing for Authorship in Different Frequency Strata“.
  Literary and Linguistic Computing 22, Nr. 1 (2007): 27–47. doi:10.1093/llc/fqi067.
- Hoover, David L. „Teasing out Authorship and Style with T-Tests and Zeta“. In Digital Humanities Conference.
  London, 2010. http://dh2010.cch.kcl.ac.uk/academic-programme/abstracts/papers/html/ab-658.html.
- Schöch, Christof. „Genre Analysis“. In Digital Humanities for Literary Studies: Theories, Methods, and Practices,
  ed. by James O’Sullivan. University Park: Pennsylvania State Univ. Press, 2017 (to appear).
"""


# =================================
# Zeta Parameters
# =================================

seglength = 1000  # int
pos = "all"  # Nc|Vv|Rg|Ag etc. depending on tagger model, or "all" if no selection
forms = "lemmata"  # words|lemmata|pos
contrast = ["subgenre", "tragicomedie", "comedie"]  # Category, Label1, Label2

# =================================
# Files and folders
# =================================
workdir = "/"  # full path to working directory; ends with slash
plaintextfolder = workdir + "text/"
metadatafile = workdir + "metadata.csv"
stoplistfile = workdir + "stoplist.txt"
taggedfolder = workdir + "tagged/"
datafolder = workdir + "data/"
resultsfolder = workdir + "results/"
contraststring = contrast[1] + "-" + contrast[2]
parameterstring = str(seglength) + "-" + forms + "-" + str(pos)

# =================================
# Functions
# =================================

# Prepare texts: tag and save (run once for a collection).
# pyzeta.prepare(plaintextfolder, taggedfolder)


# Calculate Zeta for words in two text collections
pyzeta.zeta(taggedfolder, metadatafile, contrast, datafolder, resultsfolder,
            seglength, pos, forms, stoplistfile)


# Make a nice plot with some zeta data
numwords = 20
pyzeta.plot_scores(numwords, contraststring, parameterstring, resultsfolder)


# Scatterplot of types
numfeatures = 500
cutoff = 0.30
pyzeta.plot_types(numfeatures, cutoff, contrast, contraststring, parameterstring, resultsfolder)


# Threeway comparison (NON-FUNCTIONAL)
# numfeatures = 20
# components = [1, 2]
# threecontrast = [["subgenre", "comedie", "tragedie", "tragicomedie"],
#                  ["comedie", "comedie", "other"],
#                  ["tragedie", "tragedie", "other"],
#                  ["tragicomedie", "tragicomedie", "other"]]
# pyzeta.threeway(datafolder, zetafile, numfeatures, components, plaintextfolder, metadatafile,
#                threecontrast, seglength, mode, pos, forms, stoplist)
