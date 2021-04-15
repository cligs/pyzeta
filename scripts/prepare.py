#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: prepare.py
# author: #cf
# version: 0.3.0


"""
The functions contained in this script prepare a set of plain text files for contrastive analysis. 
"""

# =================================
# Import statements
# =================================

import os
import re
import glob
import csv
import glob
import pandas as pd
import numpy as np
from collections import Counter
import itertools
import random


from sklearn.feature_extraction.text import CountVectorizer


# =================================
# Functions: make_segments
# =================================


def read_csvfile(file):
    with open(file, "r", newline="\n", encoding="utf-8") as csvfile:
        filename, ext = os.path.basename(file).split(".")
        content = csv.reader(csvfile, delimiter='\t')
        stops = ["SENT", "''", ",", "``", ":"]
        alllines = [line for line in content if len(line) == 3 and line[1] not in stops]
        return filename, alllines


def segment_files(filename, alllines, segmentlength, max_num_segments):
    segments = []
    segmentids = []
    if segmentlength == "text":
        numsegments = 1
        segment = alllines
        segmentid = filename
        segments.append(segment)
        segmentids.append(segmentid)
    else:
        numsegments = int(len(alllines) / segmentlength)
        for i in range(0, numsegments):
            segmentid = filename + "-" + "{:04d}".format(i)
            segmentids.append(segmentid)
            segment = alllines[i * segmentlength:(i + 1) * segmentlength]
            segments.append(segment)
        if max_num_segments != -1 and numsegments > max_num_segments:
            #chosen_ids = sorted(np.random.randint(0, numsegments, max_num_segments))
            chosen_ids = sorted(random.sample(range(0, numsegments), max_num_segments))
            #print(chosen_ids)
            segments = [segments[i] for i in chosen_ids]
            segmentids = [segmentids[i] for i in chosen_ids]
    return segmentids, segments


def make_segments(file, segmentfolder, segmentlength, max_num_segments=-1):
    if not os.path.exists(segmentfolder):
        os.makedirs(segmentfolder)
    filename, alllines = read_csvfile(file)
    segmentids, segments = segment_files(filename, alllines, segmentlength, max_num_segments)
    return segmentids, segments


# =================================
# Functions: select_features
# =================================

def read_stoplistfile(stoplistfile):
    with open(stoplistfile, "r", encoding="utf-8") as infile:
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
        ## TODO: Turn the test "pos in line[1]" around to allow for more than one POS tag to be defined as the filter.
        if forms == "words":
            selected = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "pos":
            selected = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "lemmata":
            selected = []
            for line in segment:
                if len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist and line[2] != "<unknown>":
                    selected.append(line[2])
                elif len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist and line[2] == "<unknown>":
                    selected.append(line[0])
    else:
        selected = []
    selected = list(selected)
    return selected


def save_segment(features, segmentfolder, segmentid):
    # TODO: remove this intermediate saving step, feed directly into make_dtm.
    segmentfile = segmentfolder + segmentid + ".txt"
    featuresjoined = " ".join(features)
    with open(segmentfile, "w", encoding="utf-8") as outfile:
        outfile.write(featuresjoined)


def select_features(segmentfolder, segmentids, segments, stoplistfile, features):
    stoplist = read_stoplistfile(stoplistfile)
    # print(len(segmentids))
    for i in range(len(segmentids)):
        segment = segments[i]
        selected = perform_selection(segment, stoplist, features)
        save_segment(selected, segmentfolder, segmentids[i])


# =================================
# Functions: make_dtm
# =================================


def read_plaintext(file):
    with open(file, "r", encoding="utf-8") as infile:
        text = infile.read().split(" ")
        features = [form for form in text if form]
        return features


#def count_features(features, filename):
#    featurecount = Counter(features)
#    featurecount = dict(featurecount)
#    featurecount = pd.Series(featurecount, name=filename)
#    #print(featurecount.loc["man"])
#    return featurecount


def save_dataframe(allfeaturecounts, dtmfolder, parameterstring):
    dtmfile = dtmfolder + "dtm_" + parameterstring + "_absolutefreqs.csv"
    #print("\nallfeaturecounts\n", allfeaturecounts.head()) 
    allfeaturecounts.to_hdf(dtmfile, key="df")
    with open(dtmfile, "w", encoding = "utf-8") as outfile:
       allfeaturecounts.to_csv(outfile, sep="\t")



def make_dtm(segmentfolder, dtmfolder, parameterstring):
    filenames = glob.glob(os.path.join(segmentfolder, "*.txt"))
    idnos = [os.path.basename(idno).split(".")[0] for idno in filenames]
    vectorizer = CountVectorizer(input='filename')
    dtm = vectorizer.fit_transform(filenames)  # a sparse matrix#
    vocab = vectorizer.get_feature_names()  # a list
    allfeaturecounts = pd.DataFrame(dtm.toarray(), columns=vocab)
    allfeaturecounts["idno"] = idnos
    allfeaturecounts.set_index("idno", inplace=True)
    #allfeaturecounts.drop("idno", inplace=True)
    allfeaturecounts = allfeaturecounts.fillna(0).astype(int)
    print("\nallfeaturecounts\n", allfeaturecounts.head())
    #save_dataframe(allfeaturecounts, dtmfolder, parameterstring)
    return allfeaturecounts


# =================================
# Functions: transform_dtm
# =================================


def read_freqsfile(filepath):
    with open(filepath, "r", newline="\n", encoding="utf-8") as csvfile:
        absolutefreqs = pd.read_csv(csvfile, sep='\t', index_col=0)
        print("\nabsolutefreqs\n", absolutefreqs.head())
        return absolutefreqs


def transform_dtm(absolutefreqs, segmentlength):
    print("Next: transforming to relative frequencies...")
    absolutefreqs_sum = pd.Series(absolutefreqs.sum(axis=1))
    print("absolutfreqs_sum", absolutefreqs_sum.values)
    if segmentlength == "text":
        relativefreqs = absolutefreqs.div(absolutefreqs_sum, axis='rows', level=None)
        print("\nrelfreqs\n", relativefreqs.head(20), segmentlength)
    else:
        relativefreqs = absolutefreqs / segmentlength
        print("\nrelfreqs\n", relativefreqs.head(), segmentlength)
    print("Next: transforming to binary frequencies...")
    binaryfreqs = absolutefreqs.copy()
    binaryfreqs[binaryfreqs > 0] = 1
    print("\nbinaryfreqs\n", binaryfreqs.head(50), segmentlength)
    print("\nabsolutefreqs\n", absolutefreqs.head(50), segmentlength)
    return absolutefreqs_sum, relativefreqs, binaryfreqs


def save_transformed(relativefreqs, binaryfreqs, dtmfolder, parameterstring):
    transformedfile = dtmfolder + "dtm_" + parameterstring + "_relativefreqs.csv"
    with open(transformedfile, "w", encoding = "utf-8") as outfile:
        relativefreqs.to_csv(outfile, sep="\t")
    #relativefreqs.to_hdf(transformedfile, key="df")
    transformedfile = dtmfolder + "dtm_" + parameterstring + "_binaryfreqs.csv"
    #binaryfreqs.to_hdf(transformedfile, key="df")
    with open(transformedfile, "w", encoding = "utf-8") as outfile:
        binaryfreqs.to_csv(outfile, sep="\t")



# =================================
# Functions: main
# =================================


def main(taggedfolder, segmentfolder, datafolder, dtmfolder, segmentlength, max_num_segments, stoplistfile, featuretype):
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    if not os.path.exists(dtmfolder):
        os.makedirs(dtmfolder)
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    print("\n--prepare")
    import shutil
    if os.path.exists(segmentfolder):
        shutil.rmtree(segmentfolder)
    counter = 0
    for file in glob.glob(taggedfolder + "*.csv"):
        filename, ext = os.path.basename(file).split(".")
        counter +=1
        print("next: file no", counter, "- file", filename)        
        segmentids, segments = make_segments(file, segmentfolder, segmentlength, max_num_segments)
        select_features(segmentfolder, segmentids, segments, stoplistfile, featuretype)
    allfeaturecounts = make_dtm(segmentfolder, dtmfolder, parameterstring)
    absolutefreqs = allfeaturecounts
    #absolutefreqs = read_freqsfile(dtmfolder + "dtm_" + parameterstring + "_absolutefreqs.csv")
    absolutefreqs_sum, relativefreqs, binaryfreqs = transform_dtm(absolutefreqs, segmentlength)
    save_transformed(relativefreqs, binaryfreqs, dtmfolder, parameterstring)
    return absolutefreqs, relativefreqs, binaryfreqs, absolutefreqs_sum
