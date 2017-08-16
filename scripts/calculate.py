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
from sklearn import preprocessing as prp


# =================================
# Functions: calculate
# =================================


def make_idlists(metadatafile, contrast):
    """
    This function creates lists of document identifiers based on the metadata.
    Depending on the contrast defined, the two lists contain various identifiers.
    """
    with open(metadatafile, "r") as infile:
        metadata = pd.DataFrame.from_csv(infile, sep=";")
        list1 = list(metadata[metadata[contrast[0]].isin([contrast[1]])].index)
        list2 = list(metadata[metadata[contrast[0]].isin([contrast[2]])].index)
        idlists = [list1, list2]
        return idlists


def filter_dtm(datafolder, idlists):
    """
    This function splits the DTM in two parts.
    Each part consists of the segments corresponding to one partition.
    Each segment is chosen based on the file id it corresponds to.
    """
    dtmfile = datafolder + "dtm_binaryfreqs.csv"
    ids1 = "|".join([id+".*" for id in idlists[0]])
    ids2 = "|".join([id+".*" for id in idlists[1]])
    with open(dtmfile, "r") as infile:
        binary = pd.DataFrame.from_csv(infile, sep="\t")
        binary1 = binary.filter(regex=ids1, axis=1)
        binary2 = binary.filter(regex=ids2, axis=1)
    dtmfile = datafolder + "dtm_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        relative = pd.DataFrame.from_csv(infile, sep="\t")
        relative1 = relative.filter(regex=ids1, axis=1)
        relative2 = relative.filter(regex=ids2, axis=1)
    return binary1, binary2, relative1, relative2


def get_indicators(binary1, binary2, relative1, relative2):
    """
    Indicators are the mean relative frequency or the document proportions,
    depending on the method chosen.   
    """
    docprops1 = np.mean(binary1, axis=1)
    docprops1 = pd.Series(docprops1, name="docprops2")
    docprops2 = np.mean(binary2, axis=1)
    docprops2 = pd.Series(docprops2, name="docprops2")
    relfreqs1 = np.mean(relative1, axis=1)*1000
    relfreqs1 = pd.Series(relfreqs1, name="relfreqs1")
    relfreqs2 = np.mean(relative2, axis=1)*1000
    relfreqs2 = pd.Series(relfreqs2, name="relfreqs2")
    return docprops1, docprops2, relfreqs1, relfreqs2


def calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2):
    """
    Scores are based on the division or substraction of the indicators.
    For division, the scores are adjusted to avoid division by zero.
    For 
    The combination of binary features and subtraction is Burrows' Zeta.
    The combination of relative features and division corresponds to
    the ratio of relative frequencies.
    """
    # Original Zeta and variants
    origzeta = docprops1 - docprops2
    origzeta = pd.Series(origzeta, name="zeta")
    # Prepare scaler to rescale variants to range of origzeta
    lowest = min(origzeta)
    highest = max(origzeta)
    scaler = prp.MinMaxScaler(feature_range=(lowest,highest))
    # Zeta with division instead of subtraction
    divzeta = (docprops1 + 0.00000000001) / (docprops2 + 0.00000000001)
    divzeta = pd.Series(divzeta, name="divzeta")
    scaler.fit(divzeta)
    divzeta = scaler.transform(divzeta)
    # Zeta with log transform of values
    logzeta = np.log(docprops1 + 0.001) - np.log(docprops2 + 0.001)
    logzeta = pd.Series(logzeta, name="logzeta")   
    scaler.fit(logzeta)
    logzeta = scaler.transform(logzeta)
    # Standard ratio of relative frequencies
    ratiorelfreqs = (relfreqs1 + 0.00000000001) / (relfreqs2 + 0.00000000001)
    ratiorelfreqs = pd.Series(ratiorelfreqs, name="ratiorelfreqs")
    scaler.fit(ratiorelfreqs)
    ratiorelfreqs = scaler.transform(ratiorelfreqs)
    # Subtraction of relative frequencies
    subrelfreqs = relfreqs1 - relfreqs2
    subrelfreqs = pd.Series(subrelfreqs, name="subrelfreqs")
    scaler.fit(subrelfreqs)
    subrelfreqs = scaler.transform(subrelfreqs)
    # Subtraction of relative frequencies after log transformation
    logrelfreqs = np.log(relfreqs1 + 0.001) - np.log(relfreqs2 + 0.001)
    logrelfreqs = pd.Series(logrelfreqs, name="logrelfreqs")   
    scaler.fit(logrelfreqs)
    logrelfreqs = scaler.transform(logrelfreqs)
    return origzeta, divzeta, logzeta, ratiorelfreqs, subrelfreqs, logrelfreqs
    

def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, origzeta, divzeta, logzeta, ratiorelfreqs, subrelfreqs, logrelfreqs):
    results = pd.DataFrame({
    "docprops1":docprops1,
    "docprops2":docprops2,
    "relfreqs1":relfreqs1,
    "relfreqs2":relfreqs2,
    "origzeta":origzeta,
    "divzeta":divzeta,
    "logzeta":logzeta,
    "ratiorelfreqs":ratiorelfreqs,
    "subrelfreqs":subrelfreqs,
    "logrelfreqs":logrelfreqs})
    print(results.columns.tolist())
    results = results[["docprops1", "docprops2", "origzeta", "logzeta", "divzeta", "relfreqs1", "relfreqs2", "ratiorelfreqs", "subrelfreqs", "logrelfreqs"]]
    results.sort_values(by="origzeta", ascending=False, inplace=True)
    print(results.head(10), "\n", results.tail(10))
    return results
    

def save_results(results, resultsfolder):
    resultsfile = resultsfolder + "zetaresults.csv"
    with open(resultsfile, "w") as outfile:
        results.to_csv(outfile, sep="\t")


# =================================
# Function: main
# =================================


def main(datafolder, metadatafile, contrast, resultsfolder):
    print("--calculate")
    idlists = make_idlists(metadatafile, contrast)
    binary1, binary2, relative1, relative2 = filter_dtm(datafolder, idlists)
    docprops1, docprops2, relfreqs1, relfreqs2 = get_indicators(binary1, binary2, relative1, relative2)
    origzeta, divzeta, logzeta, ratiorelfreqs, subrelfreqs, logrelfreqs = calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, origzeta, divzeta, logzeta, ratiorelfreqs, subrelfreqs, logrelfreqs)
    save_results(results, resultsfolder)
    

    
    
