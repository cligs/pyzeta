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
import random


# =================================
# Functions: calculate
# =================================


def make_idlists(metadatafile, separator, contrast):
    """
    This function creates lists of document identifiers based on the metadata.
    Depending on the contrast defined, the two lists contain various identifiers.
    """
    with open(metadatafile, "r") as infile:
        metadata = pd.DataFrame.from_csv(infile, sep=separator)
        #print(metadata.head())
        if contrast[0] != "random":
            list1 = list(metadata[metadata[contrast[0]].isin([contrast[1]])].index)
            list2 = list(metadata[metadata[contrast[0]].isin([contrast[2]])].index)
        elif contrast[0] == "random":
            allidnos = list(metadata.loc[:, "idno"])
            allidnos = random.sample(allidnos, len(allidnos))
            list1 = allidnos[:int(len(allidnos)/2)]
            list2 = allidnos[int(len(allidnos)/2):]
            print(list1[0:5])
            print(list2[0:5])
        idlists = [list1, list2]
        return idlists


def filter_dtm(datafolder, parameterstring, idlists):
    """
    This function splits the DTM in two parts.
    Each part consists of the segments corresponding to one partition.
    Each segment is chosen based on the file id it corresponds to.
    """
    dtmfile = datafolder + "dtm_"+parameterstring+"_binaryfreqs.csv"
    ids1 = "|".join([id+".*" for id in idlists[0]])
    ids2 = "|".join([id+".*" for id in idlists[1]])
    with open(dtmfile, "r") as infile:
        binary = pd.DataFrame.from_csv(infile, sep="\t")
        binary1 = binary.filter(regex=ids1, axis=1)
        binary2 = binary.filter(regex=ids2, axis=1)
    dtmfile = datafolder + "dtm_"+parameterstring+"_relativefreqs.csv"
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
    docprops1 = pd.Series(docprops1, name="docprops1")
    docprops2 = np.mean(binary2, axis=1)
    docprops2 = pd.Series(docprops2, name="docprops2")
    relfreqs1 = np.mean(relative1, axis=1)*1000
    relfreqs1 = pd.Series(relfreqs1, name="relfreqs1")
    relfreqs2 = np.mean(relative2, axis=1)*1000
    relfreqs2 = pd.Series(relfreqs2, name="relfreqs2")
    return docprops1, docprops2, relfreqs1, relfreqs2


def calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, logaddition):
    """
    This function implements several variants of Zeta by modifying some key parameters.
    Scores can be document proportions (binary features) or relative frequencies.
    Scores can be taken directly or subjected to a log-transformation (log2, log10)
    Scores can be subtracted from each other or divided by one another.
    The combination of document proportion, no transformation and subtraction is Burrows' Zeta.
    The combination of relative frequencies, no transformation, and division corresponds to
    the ratio of relative frequencies.
    """
    # Define logaddition and division-by-zero avoidance addition
    logaddition = logaddition
    divaddition = 0.00000000001
    # == Calculate subtraction variants ==
    # sd0 - Subtraction, docprops, untransformed a.k.a. "original Zeta"
    sd0 = docprops1 - docprops2
    sd0 = pd.Series(sd0, name="sd0")
    # Prepare scaler to rescale variants to range of sd0 (original Zeta)
    scaler = prp.MinMaxScaler(feature_range=(min(sd0),max(sd0)))
    # sd2 - Subtraction, docprops, log2-transformed
    sd2 = np.log2(docprops1 + logaddition) - np.log2(docprops2 + logaddition)
    sd2 = pd.Series(sd2, name="sd2")
    sd2 = scaler.fit_transform(sd2.values.reshape(-1, 1))
    # sdX - Subtraction, docprops, log10-transformed
    sdX = np.log10(docprops1 + logaddition) - np.log10(docprops2 + logaddition)
    sdX = pd.Series(sdX, name="sdX")
    sdX = scaler.fit_transform(sdX.values.reshape(-1, 1))
    # sr0 - Subtraction, relfreqs, untransformed
    sr0 = relfreqs1 - relfreqs2
    sr0 = pd.Series(sr0, name="sr0")
    sr0 = scaler.fit_transform(sr0.values.reshape(-1, 1))
    # sr2 - Subtraction, relfreqs, log2-transformed
    sr2 = np.log2(relfreqs1 + logaddition) - np.log2(relfreqs2 + logaddition)
    sr2 = pd.Series(sr2, name="sr2")
    sr2 = scaler.fit_transform(sr2.values.reshape(-1, 1))
    # srX - Subtraction, relfreqs, log10-transformed
    srX = np.log10(relfreqs1 + logaddition) - np.log10(relfreqs2 + logaddition)
    srX = pd.Series(srX, name="srX")
    srX = scaler.fit_transform(srX.values.reshape(-1, 1))

    # == Division variants ==
    # dd0 - Division, docprops, untransformed
    dd0 = (docprops1 + divaddition) / (docprops2 + divaddition)
    dd0 = pd.Series(dd0, name="dd0")
    dd0 = scaler.fit_transform(dd0.values.reshape(-1, 1))
    # dd2 - Division, docprops, log2-transformed
    dd2 = np.log2(docprops1 + logaddition) / np.log2(docprops2 + logaddition)
    dd2 = pd.Series(dd2, name="dd2")
    dd2 = scaler.fit_transform(dd2.values.reshape(-1, 1))
    # ddX - Division, docprops, log10-transformed
    ddX = np.log10(docprops1 + logaddition) / np.log10(docprops2 + logaddition)
    ddX = pd.Series(ddX, name="ddX")
    ddX = scaler.fit_transform(ddX.values.reshape(-1, 1))
    # dr0 - Division, relfreqs, untransformed
    dr0 = (relfreqs1 + divaddition) / (relfreqs2 + divaddition)
    dr0 = pd.Series(dr0, name="dr0")
    dr0 = scaler.fit_transform(dr0.values.reshape(-1, 1))
    # dr2 - Division, relfreqs, log2-transformed
    dr2 = np.log2(relfreqs1 + logaddition) / np.log2(relfreqs2 + logaddition)
    dr2 = pd.Series(dr2, name="dr2")
    dr2 = scaler.fit_transform(dr2.values.reshape(-1, 1))
    # drX - Division, relfreqs, log10-transformed
    drX = np.log10(relfreqs1 + logaddition) / np.log10(relfreqs2 + logaddition)
    drX = pd.Series(drX, name="drX")
    drX = scaler.fit_transform(drX.values.reshape(-1, 1))

    # Return all zeta variant scores
    return sd0, sd2.flatten(), sdX.flatten(), sr0.flatten(), sr2.flatten(), srX.flatten(), dd0.flatten(), dd2.flatten(), ddX.flatten(), dr0.flatten(), dr2.flatten(), drX.flatten()


def get_meanrelfreqs(datafolder, parameterstring):
    dtmfile = datafolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        meanrelfreqs = pd.DataFrame.from_csv(infile, sep="\t")
        meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
        #print(meanrelfreqs.head(100))
        return meanrelfreqs


def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs,
    sd0, sd2, sdX, sr0, sr2, srX, dd0, dd2, ddX, dr0, dr2, drX):
    results = pd.DataFrame({
    "docprops1" : docprops1,
    "docprops2" : docprops2,
    "relfreqs1" : relfreqs1,
    "relfreqs2" : relfreqs2,
    "meanrelfreqs" :meanrelfreqs,
    "sd0" : sd0,
    "sd2" : sd2,
    "sdX" : sdX,
    "sr0" : sr0,
    "sr2" : sr2,
    "srX" : srX,
    "dd0" : dd0,
    "dd2" : dd2,
    "ddX" : ddX,
    "dr0" : dr0,
    "dr2" : dr2,
    "drX" : drX})
    #print(results.columns.tolist())
    results = results[[
    "docprops1",
    "docprops2",
    "relfreqs1",
    "relfreqs2",
    "meanrelfreqs",
    "sd0",
    "sd2",
    "sdX",
    "sr0",
    "sr2",
    "srX",
    "dd0",
    "dd2",
    "ddX",
    "dr0",
    "dr2",
    "drX"]]
    results.sort_values(by="sd0", ascending=False, inplace=True)
    #print(results.head(10), "\n", results.tail(10))
    return results


def save_results(results, resultsfile):
    with open(resultsfile, "w") as outfile:
        results.to_csv(outfile, sep="\t")


# =================================
# Function: main
# =================================


def main(datafolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype):
    print("--calculate")
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    idlists = make_idlists(metadatafile, separator, contrast)
    binary1, binary2, relative1, relative2 = filter_dtm(datafolder, parameterstring, idlists)
    docprops1, docprops2, relfreqs1, relfreqs2 = get_indicators(binary1, binary2, relative1, relative2)
    sd0, sd2, sdX, sr0, sr2, srX, dd0, dd2, ddX, dr0, dr2, drX = calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, logaddition)
    meanrelfreqs = get_meanrelfreqs(datafolder, parameterstring)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs,
    sd0, sd2, sdX, sr0, sr2, srX, dd0, dd2, ddX, dr0, dr2, drX)
    save_results(results, resultsfile)




