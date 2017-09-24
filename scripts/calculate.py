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


def filter_dtm(dtmfolder, parameterstring, idlists):
    """
    This function splits the DTM in two parts.
    Each part consists of the segments corresponding to one partition.
    Each segment is chosen based on the file id it corresponds to.
    """
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_binaryfreqs.csv"
    ids1 = "|".join([id+".*" for id in idlists[0]])
    ids2 = "|".join([id+".*" for id in idlists[1]])
    with open(dtmfile, "r") as infile:
        binary = pd.DataFrame.from_csv(infile, sep="\t")
        #print(binary.head())
        binary1 = binary.filter(regex=ids1, axis=1)
        binary2 = binary.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        relative = pd.DataFrame.from_csv(infile, sep="\t")
        relative1 = relative.filter(regex=ids1, axis=1)
        relative2 = relative.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_absolutefreqs.csv"
    with open(dtmfile, "r") as infile:
        absolute = pd.DataFrame.from_csv(infile, sep="\t")
        absolute1 = absolute.filter(regex=ids1, axis=1)
        absolute2 = absolute.filter(regex=ids2, axis=1)
    return binary1, binary2, relative1, relative2, absolute1, absolute2


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


def calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, absolute1, absolute2, logaddition, segmentlength, idlists):
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

    # Calculate Gries "deviation of proportions" (DP)
    segnum1 = len(absolute1.columns.values)
    segnum2 = len(absolute2.columns.values)
    seglens1 = [segmentlength] * segnum1 
    seglens2 = [segmentlength] * segnum2
    crpsize1 = sum(seglens1)
    crpsize2 = sum(seglens2)
    #print("segments", segnum1, segnum2)
    totalfreqs1 = np.sum(absolute1, axis=1)
    totalfreqs2 = np.sum(absolute2, axis=1)
    #print("totalfreqs", totalfreqs1, totalfreqs2)
    expprops1 = np.array(seglens1) / crpsize1
    expprops2 = np.array(seglens2) / crpsize2
    #print("exprops", expprops1, expprops2)
    #print(absolute1.head())
    #print(totalfreqs1)
    obsprops1 = absolute1.div(totalfreqs1, axis=0)
    obsprops1 = obsprops1.fillna(expprops1[0]) # was: expprops1[0]
    obsprops2 = absolute2.div(totalfreqs2, axis=0)
    obsprops2 = obsprops2.fillna(expprops2[0]) # was: expprops2[0]
    devprops1 = (np.sum(abs(expprops1 - obsprops1), axis=1) /2 )
    devprops2 = (np.sum(abs(expprops2 - obsprops2), axis=1) /2 )
    #print(devprops1.head())
    #print(devprops2.head())

    # Calculate DP variants ("g" for Gries)
    sg0 = devprops1 - devprops2
    sg0 = pd.Series(sg0, name="sg0")
    sg0 = scaler.fit_transform(sg0.values.reshape(-1, 1))
    sg2 = np.log2(devprops1 + logaddition) - np.log2(devprops2 + logaddition)
    sg2 = pd.Series(sg2, name="sg2")
    sg2 = scaler.fit_transform(sg2.values.reshape(-1, 1))
    dg0 = (devprops1 + divaddition) / (devprops2 + divaddition)
    dg0 = pd.Series(dg0, name="dg0")
    dg0 = scaler.fit_transform(dg0.values.reshape(-1, 1))
    dg2 = np.log2(devprops1 + logaddition) / np.log2(devprops2 + logaddition)
    dg2 = pd.Series(dg2, name="dg2")
    dg2 = scaler.fit_transform(dg2.values.reshape(-1, 1))

    # Return all zeta variant scores
    return sd0, sd2.flatten(), sr0.flatten(), sr2.flatten(), sg0.flatten(), sg2.flatten(), dd0.flatten(), dd2.flatten(), dr0.flatten(), dr2.flatten(), dg0.flatten(), dg2.flatten(), devprops1, devprops2


def get_meanrelfreqs(dtmfolder, parameterstring):
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        meanrelfreqs = pd.DataFrame.from_csv(infile, sep="\t")
        meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
        #print(meanrelfreqs.head(100))
        return meanrelfreqs


def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs, sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2):
    results = pd.DataFrame({
    "docprops1" : docprops1,
    "docprops2" : docprops2,
    "relfreqs1" : relfreqs1,
    "relfreqs2" : relfreqs2,
    "devprops1" : devprops1,
    "devprops2" : devprops2,
    "meanrelfreqs" :meanrelfreqs,
    "sd0" : sd0,
    "sd2" : sd2,
    "sr0" : sr0,
    "sr2" : sr2,
    "sg0" : sg0,
    "sg2" : sg2,
    "dd0" : dd0,
    "dd2" : dd2,
    "dr0" : dr0,
    "dr2" : dr2,
    "dg0" : dg0,
    "dg2" : dg2})
    #print(results.columns.tolist())
    results = results[[
    "docprops1",
    "docprops2",
    "relfreqs1",
    "relfreqs2",
    "devprops1",
    "devprops2",
    "meanrelfreqs",
    "sd0",
    "sd2",
    "sr0",
    "sr2",
    "sg0",
    "sg2",
    "dd0",
    "dd2",
    "dr0",
    "dr2",
    "dg0",
    "dg2"]]
    results.sort_values(by="sg0", ascending=False, inplace=True)
    #print(results.head(10), "\n", results.tail(10))
    return results


def save_results(results, resultsfile):
    with open(resultsfile, "w") as outfile:
        results.to_csv(outfile, sep="\t")


# =================================
# Function: main
# =================================


def main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype):
    print("--calculate")
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    idlists = make_idlists(metadatafile, separator, contrast)
    binary1, binary2, relative1, relative2, absolute1, absolute2 = filter_dtm(dtmfolder, parameterstring, idlists)
    docprops1, docprops2, relfreqs1, relfreqs2 = get_indicators(binary1, binary2, relative1, relative2)
    sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2, devprops1, devprops2 = calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, absolute1, absolute2, logaddition, segmentlength, idlists)
    meanrelfreqs = get_meanrelfreqs(dtmfolder, parameterstring)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs,
    sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2)
    save_results(results, resultsfile)




