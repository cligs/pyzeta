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

Currently, the following processes are supported:
- Prepare a text collection by tagging it using TreeTagger (pyzeta.prepare; run once per collection)
- For any two partitions of the collection, create a matrix of per-segment type frequencies
  and calculate the zeta scores for the vocabulary (pyzeta.zeta).
  There are options to choose word forms or lemmata or POS as features.
  There is the possibility to filter features based on their POS.
- Visualize the most distinctive words as an interactive, horizontal bar chart. (pyzeta.plot_scores)
- Visualize the feature distribution as an interactive scatterplot (pyzeta.plot_types)
- Visualize a threeway comparison of the proportions of the top-distinctive features (pyzeta.threeway)

The script expects the following as input:
- A folder with plain text files
- A metadata file with category information about each file, identified through "idno" = filename
- A file with stopwords, one per line

You can set the following parameters:
- How many word forms should each segment have (500-5000 may be reasonable)
- Which POS should be selected? "all" selects all, "Xy" will select words corresponding to Xy POS tag.
- Which partitions of the data should be contrasted? Indicate the category and the two contrasting labels.

Requirements, installation and usage
- Requirements: Linux OS, Python 3 with pandas, numpy, treetaggerwrapper and pygal
- Installation: Simply copy pyzeta.py and run_pyzeta.py to a common location on your computer
- Usage: Open pyzeta.py and run_pyzeta.py with an IDE such as Spyder or PyCharm, adapt the parameters in run_pyzeta.py and run from the IDE.

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

seglength = 3000  # int; 1000|2000|5000 are reasonable
pos = "all"  # Nc|Vv|Rg|Ag etc. depending on tagger model, or "all" if no selection
forms = "lemmata"  # words|lemmata|pos
contrast = ["", "", ""]  # category, label1, label2

# =================================
# Files and folders
# =================================
workdir = "/"  # full path to working directory; ends with slash
plaintextfolder = os.path.join(workdir, "text", "")
metadatafile = os.path.join(workdir, "metadata.csv")
stoplistfile = os.path.join(workdir, "stoplist.txt")
taggedfolder = os.path.join(workdir, "tagged", "")
datafolder = os.path.join(workdir, "data", "")
resultsfolder = os.path.join(workdir, "results", "")
contraststring = contrast[1] + "-" + contrast[2]
parameterstring = str(seglength) + "-" + forms + "-" + str(pos)


# =================================
# Functions
# =================================

# Prepare texts: tag and save (run once for a collection).
# language = "fr"  # TreeTagger language model code
pyzeta.prepare(plaintextfolder, language, taggedfolder)


# Calculate Zeta for words in two text collections
pyzeta.zeta(taggedfolder, metadatafile, contrast, datafolder, resultsfolder,
            seglength, pos, forms, stoplistfile)


# Barchart with the most extreme zeta values
numfeatures = 25
pyzeta.plot_zetascores(numfeatures, contrast, contraststring, parameterstring, resultsfolder)


# Scatterplot of types
numfeatures = 200  # int
cutoff = 0.40
pyzeta.plot_types(numfeatures, cutoff, contrast, contraststring, parameterstring, resultsfolder)


# Threeway comparison (simple)
numfeatures = 25  # int
thirdgroup = ["", ""]  # category, label3
sortby = "comedie"  # label
mode = "generate"  # string; generate|analyze
pyzeta.threeway(datafolder, resultsfolder, contrast, contraststring, parameterstring,
                thirdgroup, numfeatures, sortby, mode)
