#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: preprocess.py
# author: #cf
# version: 0.3.0

"""
The "preprocess" module is the first step in the pyzeta pipeline.
This module deals with linguistic annotation of the texts.
Subsequent modules are: prepare, calculate and visualize.
"""


# =================================
# Import statements
# =================================

import os
import re
import csv
import glob
import pandas as pd
import numpy as np
from collections import Counter
import treetaggerwrapper


# =================================
# Functions
# =================================


def read_plaintext(file):
    with open(file, "r") as infile:
        text = infile.read()
        return text


def run_treetagger(text, language):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG=language)
    tagged = tagger.tag_text(text)
    return tagged


def save_tagged(taggedfolder, filename, tagged):
    taggedfilename = taggedfolder + "/" + filename + ".csv"
    with open(taggedfilename, "w") as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for item in tagged:
            item = re.split("\t", item)
            writer.writerow(item)


# =================================
# Functions: preprocess
# =================================


def main(plaintextfolder, taggedfolder, language):
    print("--preprocess")
    if not os.path.exists(taggedfolder):
        os.makedirs(taggedfolder)
    for file in glob.glob(plaintextfolder + "*.txt"):
        filename, ext = os.path.basename(file).split(".")
        text = read_plaintext(file)
        tagged = run_treetagger(text, language)
        save_tagged(taggedfolder, filename, tagged)

