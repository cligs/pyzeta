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
    """
    Reads a plain text file. 
    Returns the text content as a string.
    """
    with open(file, "r") as infile:
        text = infile.read()
        text = re.sub("’", "'", text)
        return text


def run_treetagger(text, language):
    """
    Runs treetagger on the text string. 
    Returns a treetagger tagged object. 
    """
    tagger = treetaggerwrapper.TreeTagger(TAGLANG=language)
    tagged = tagger.tag_text(text)
    return tagged


def save_tagged(taggedfolder, filename, tagged):
    """
    Takes the treetagger output and writes it to a CSV file.
    """
    taggedfilename = taggedfolder + "/" + filename + ".csv"
    with open(taggedfilename, "w") as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for item in tagged:
            item = re.split("\t", item)
            writer.writerow(item)


def sanity_check(text, tagged): 
    """
    Performs a simple sanity check on the data. 
    Checks number of words in inpu text. 
    Checks number of lines in tagged output. 
    If these numbers are similar, it looks good. 
    """
    text = re.sub("([,.:;!?])", " \1", text)
    text = re.split("\s+", text)
    print("number of words", len(text)) 
    print(text[0:10])
    print("number of lines", len(tagged))
    print(tagged[0:10])
    if len(tagged) == 0: 
        print("Sanity check: Tagging error: nothing tagged.")
    elif len(tagged) / len(text) < 0.8  or len(tagged) / len(text) > 1.2: 
        print("Sanity check: Tagging error: strong length difference.")
    else: 
        print("Sanity check: Tagging seems to have worked.")


# =================================
# Functions: preprocess
# =================================


def main(plaintextfolder, taggedfolder, language, sanitycheck):
    print("\n--preprocess")
    if not os.path.exists(taggedfolder):
        os.makedirs(taggedfolder)
    counter = 0
    for file in glob.glob(plaintextfolder + "*.txt"):
        filename, ext = os.path.basename(file).split(".")
        counter +=1
        print("next: file", counter, ":", filename)
        text = read_plaintext(file)
        tagged = run_treetagger(text, language)
        save_tagged(taggedfolder, filename, tagged)
        if sanitycheck == "yes": 
            sanity_check(text, tagged)

