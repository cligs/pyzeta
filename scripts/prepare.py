#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: prepare.py
# author: #cf
# version: 0.3.0


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
import itertools


# =================================
# Functions: make_segments
# =================================


def read_csvfile(file):
    with open(file, "r", newline="\n") as csvfile:
        filename, ext = os.path.basename(file).split(".")
        content = csv.reader(csvfile, delimiter='\t')
        stops = ["SENT", "''", ",", "``", ":"]
        alllines = [line for line in content if len(line)==3 and line[1] not in stops]
        return filename, alllines


def segment_files(filename, alllines, segmentlength):
    numsegments = int(len(alllines) / segmentlength)
    segments = []
    segmentids = []
    for i in range(0, numsegments):
        segmentid = filename + "-" + "{:04d}".format(i)
        segmentids.append(segmentid)
        segment = alllines[i * segmentlength:(i + 1) * segmentlength]
        segments.append(segment)
    return segmentids, segments


def make_segments(file, segmentfolder, segmentlength):
    if not os.path.exists(segmentfolder):
        os.makedirs(segmentfolder)
    filename, alllines = read_csvfile(file)
    segmentids, segments = segment_files(filename, alllines, segmentlength)
    return segmentids, segments 
        

# =================================
# Functions: select_features
# =================================


def read_stoplistfile(stoplistfile):
    with open(stoplistfile, "r") as infile:
        stoplist = infile.read()
        stoplist = list(re.split("\n", stoplist))
        return stoplist


def perform_selection(segment, stoplist, featuretype):
    """
    Selects the desired features (words, lemmas or pos) from each segment of text.
    TODO: Add a replacement feature for words like "j'" or "-ils"
    """
    pos = featuretype[1]
    forms = featuretype[0]
    if pos == "all":
        if forms == "words":
            selected = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and line[2] not in stoplist]
        elif forms == "lemmata":
            selected = [line[2].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and line[2] not in stoplist]
        elif forms == "pos":
            selected = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and line[2] not in stoplist]
    elif pos != "all":
        if forms == "words":
            selected = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "lemmata":
            selected = [line[2].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "pos":
            features = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
    else:
        selected = []
    selected = list(selected)
    return selected


def save_segment(features, segmentfolder, segmentid):
    # TODO: remove this intermediate saving step, feed directly into make_dtm.
    segmentfile = segmentfolder + segmentid + ".txt"
    featuresjoined = " ".join(features)
    with open(segmentfile, "w") as outfile:
        outfile.write(featuresjoined)


def select_features(segmentfolder, segmentids, segments, stoplistfile, features):
    stoplist = read_stoplistfile(stoplistfile)
    #print(len(segmentids))
    for i in range(len(segmentids)):
        segment = segments[i]
        selected = perform_selection(segment, stoplist, features)
        save_segment(selected, segmentfolder, segmentids[i])
        
    
    
# =================================
# Functions: make_dtm
# =================================


def read_plaintext(file):
    with open(file, "r") as infile:
        text = infile.read().split(" ")
        features = [form for form in text if form]
        return features
        

def count_features(features, filename):
    featurecount = Counter(features)
    featurecount = dict(featurecount)
    featurecount = pd.Series(featurecount, name=filename)
    #print(featurecount.loc["man"])
    return featurecount


def save_dataframe(allfeaturecounts, datafolder):
    dtmfile = datafolder + "dtm_absolute-freqs.csv"
    with open(dtmfile, "w") as outfile:
        allfeaturecounts.to_csv(outfile, sep="\t")


def make_dtm(segmentfolder, datafolder):
    allfeaturecounts = []
    for file in glob.glob(segmentfolder + "*.txt"):
        features = read_plaintext(file)
        filename, ext = os.path.basename(file).split(".")
        featurecount = count_features(features, filename)
        allfeaturecounts.append(featurecount)
    allfeaturecounts = pd.concat(allfeaturecounts, axis=1)
    allfeaturecounts = allfeaturecounts.fillna(0).astype(int)
    save_dataframe(allfeaturecounts, datafolder)
    

# =================================
# Functions: transform_dtm
# =================================


def read_freqsfile(filepath):
    with open(filepath, "r", newline="\n") as csvfile:
        absolutefreqs = pd.DataFrame.from_csv(csvfile, sep='\t')
        return absolutefreqs


def transform_dtm(absolutefreqs, segmentlength):
    relativefreqs = absolutefreqs / segmentlength
    absolutefreqs[absolutefreqs > 0]  = 1
    binaryfreqs = absolutefreqs
    return relativefreqs, binaryfreqs


def save_transformed(relativefreqs, binaryfreqs, datafolder):
    transformedfile = datafolder + "dtm_relativefreqs.csv"
    with open(transformedfile, "w") as outfile:
        relativefreqs.to_csv(outfile, sep="\t")
    transformedfile = datafolder + "dtm_binaryfreqs.csv"
    with open(transformedfile, "w") as outfile:
        binaryfreqs.to_csv(outfile, sep="\t")


# =================================
# Functions: main
# =================================

    
def main(taggedfolder, segmentfolder, datafolder, segmentlength, stoplistfile, featuretype):
    print("--prepare")
    for file in glob.glob(taggedfolder+"*.csv"):
        segmentids, segments = make_segments(file, segmentfolder, segmentlength)
        select_features(segmentfolder, segmentids, segments, stoplistfile, featuretype)
        make_dtm(segmentfolder, datafolder)
    absolutefreqs = read_freqsfile(datafolder + "dtm_absolute-freqs.csv")
    relativefreqs, binaryfreqs = transform_dtm(absolutefreqs, segmentlength)
    save_transformed(relativefreqs, binaryfreqs, datafolder)
