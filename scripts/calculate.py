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
        metadata = pd.read_csv(infile, sep=separator)
        print("\nmetadata\n", metadata.head())
        #metadata = metadata.drop("Unnamed: 0", axis=1)
        metadata.set_index("idno", inplace=True)
        print("\nmetadata\n", metadata.head())
        if contrast[0] != "random":
            list1 = list(metadata[metadata[contrast[0]].isin([contrast[1]])].index)
            list2 = list(metadata[metadata[contrast[0]].isin([contrast[2]])].index)
            #print("list1", list1)
            #print("list2", list2)
        elif contrast[0] == "random":
            allidnos = list(metadata.loc[:, "idno"])
            allidnos = random.sample(allidnos, len(allidnos))
            list1 = allidnos[:int(len(allidnos)/2)]
            list2 = allidnos[int(len(allidnos)/2):]
            #print(list1[0:5])
            #print(list2[0:5])
        idlists = [list1, list2]
        #print(idlists)
        return idlists


def filter_dtm(dtmfolder, parameterstring, idlists, absolutefreqs, relativefreqs, binaryfreqs):
    """
    This function splits the DTM in two parts.
    Each part consists of the segments corresponding to one partition.
    Each segment is chosen based on the file id it corresponds to.
    """
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_binaryfreqs.csv"
    #print(dtmfile)
    #print(idlists)
    ids1 = "|".join([str(idno)+".*" for idno in idlists[0]])
    print(ids1)
    ids2 = "|".join([str(id)+".*" for id in idlists[1]])
    #print(ids2)
    binary = binaryfreqs
    relative = relativefreqs
    absolute = absolutefreqs
    binary1 = binary.T.filter(regex=ids1, axis=1)
    binary2 = binary.T.filter(regex=ids2, axis=1)
    relative1 = relative.T.filter(regex=ids1, axis=1)
    relative2 = relative.T.filter(regex=ids2, axis=1)
    absolute1 = absolute.T.filter(regex=ids1, axis=1)
    absolute2 = absolute.T.filter(regex=ids2, axis=1)
    print("\nbinary1\n", binary1.head())
    """
    with open(dtmfile, "r") as infile:
        binary = pd.read_hdf(infile, sep="\t", index_col="idno")
        print("\nbinary\n", binary.head())
        binary1 = binary.T.filter(regex=ids1, axis=1)
        print("\nbinary1\n", binary1.head())
        binary2 = binary.T.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        relative = pd.read_hdf(infile, sep="\t", index_col="idno")
        #print("\nrelative\n", relative.head())
        relative1 = relative.T.filter(regex=ids1, axis=1)
        relative2 = relative.T.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_absolutefreqs.csv"
    with open(dtmfile, "r") as infile:
        absolute = pd.read_hdf(infile, sep="\t", index_col="idno")
        #print("\nabsolute\n", absolute.head())
        absolute1 = absolute.T.filter(regex=ids1, axis=1)
        absolute2 = absolute.T.filter(regex=ids2, axis=1)
    """
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
    print("\ndocprops1\n", docprops1.head(20))
    print("\ndocprops2\n", docprops2.head(20))
    print("\nrelfreqs1\n", relfreqs1.head())
    print("\nrelfreqs2\n", relfreqs2.head())
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
    print("---calculate scores: 1/4")
    # Define logaddition and division-by-zero avoidance addition
    logaddition = logaddition+1
    divaddition = 0.00000000001
    # == Calculate subtraction variants ==
    # sd0 - Subtraction, docprops, untransformed a.k.a. "original Zeta"
    sd0 = docprops1 - docprops2
    sd0 = pd.Series(sd0, name="sd0")
    print("\nsd0\n", sd0.head(10))
    # Prepare scaler to rescale variants to range of sd0 (original Zeta)
    scaler = prp.MinMaxScaler(feature_range=(min(sd0),max(sd0)))
    # sd2 - Subtraction, docprops, log2-transformed
    sd2 = np.log2(docprops1 + logaddition) - np.log2(docprops2 + logaddition)
    sd2 = pd.Series(sd2, name="sd2")
    sd2_index = sd2.index
    sd2 = scaler.fit_transform(sd2.values.reshape(-1, 1))
    sd2 = [value[0] for value in sd2]
    sd2 = pd.Series(data=sd2, index=sd2_index)
    #print("\nsd2\n", sd2.head())
    # sdX - Subtraction, docprops, log10-transformed
    sdX = np.log10(docprops1 + logaddition) - np.log10(docprops2 + logaddition)
    sdX = pd.Series(sdX, name="sdX")
    sdX_index = sdX.index
    sdX = scaler.fit_transform(sdX.values.reshape(-1, 1))
    sdX = [value[0] for value in sdX]
    sdX = pd.Series(data=sdX, index=sdX_index)
    #print("\nsdX\n", sdX.head()) 
    # sr0 - Subtraction, relfreqs, untransformed
    sr0 = relfreqs1 - relfreqs2
    sr0 = pd.Series(sr0, name="sr0")
    sr0_index = sr0.index
    sr0 = scaler.fit_transform(sr0.values.reshape(-1, 1))
    sr0 = [value[0] for value in sr0]
    sr0 = pd.Series(data=sr0, index=sr0_index)
    # sr2 - Subtraction, relfreqs, log2-transformed
    sr2 = np.log2(relfreqs1 + logaddition) - np.log2(relfreqs2 + logaddition)
    sr2 = pd.Series(sr2, name="sr2")
    sr2_index = sr2.index
    sr2 = scaler.fit_transform(sr2.values.reshape(-1, 1))
    sr2 = [value[0] for value in sr2]
    sr2 = pd.Series(data=sr2, index=sr2_index)
    # srX - Subtraction, relfreqs, log10-transformed
    srX = np.log10(relfreqs1 + logaddition) - np.log10(relfreqs2 + logaddition)
    srX = pd.Series(srX, name="srX")
    srX_index = srX.index
    srX = scaler.fit_transform(srX.values.reshape(-1, 1))
    srX = [value[0] for value in srX]
    srX = pd.Series(data=srX, index=srX_index)

    # == Division variants ==
    print("---calculate scores: 2/4")
    # dd0 - Division, docprops, untransformed
    dd0 = (docprops1 + divaddition) / (docprops2 + divaddition)
    dd0 = pd.Series(dd0, name="dd0")
    dd0_index = dd0.index
    dd0 = scaler.fit_transform(dd0.values.reshape(-1, 1))
    dd0 = [value[0] for value in dd0]
    dd0 = pd.Series(data=dd0, index=dd0_index)
    # dd2 - Division, docprops, log2-transformed
    dd2 = np.log2(docprops1 + logaddition) / np.log2(docprops2 + logaddition)
    dd2 = pd.Series(dd2, name="dd2")
    dd2_index = dd2.index
    dd2 = scaler.fit_transform(dd2.values.reshape(-1, 1))
    dd2 = [value[0] for value in dd2]
    dd2 = pd.Series(data=dd2, index=dd2_index)
    # ddX - Division, docprops, log10-transformed
    ddX = np.log10(docprops1 + logaddition) / np.log10(docprops2 + logaddition)
    ddX = pd.Series(ddX, name="ddX")
    ddX_index = ddX.index
    ddX = scaler.fit_transform(ddX.values.reshape(-1, 1))
    ddX = [value[0] for value in ddX]
    ddX = pd.Series(data=ddX, index=ddX_index)
    # dr0 - Division, relfreqs, untransformed
    dr0 = (relfreqs1 + divaddition) / (relfreqs2 + divaddition)
    dr0 = pd.Series(dr0, name="dr0")
    dr0_index = dr0.index
    dr0 = scaler.fit_transform(dr0.values.reshape(-1, 1))
    dr0 = [value[0] for value in dr0]
    dr0 = pd.Series(data=dr0, index=dr0_index)
    # dr2 - Division, relfreqs, log2-transformed
    dr2 = np.log2(relfreqs1 + logaddition) / np.log2(relfreqs2 + logaddition)
    dr2 = pd.Series(dr2, name="dr2")
    dr2_index = dr2.index
    dr2 = scaler.fit_transform(dr2.values.reshape(-1, 1))
    dr2 = [value[0] for value in dr2]
    dr2 = pd.Series(data=dr2, index=dr2_index)
    # drX - Division, relfreqs, log10-transformed
    drX = np.log10(relfreqs1 + logaddition) / np.log10(relfreqs2 + logaddition)
    drX = pd.Series(drX, name="drX")
    drX_index = drX.index
    drX = scaler.fit_transform(drX.values.reshape(-1, 1))
    drX = [value[0] for value in drX]
    drX = pd.Series(data=drX, index=drX_index)

    # Calculate Gries "deviation of proportions" (DP)
    print("---calculate scores: 3/4")
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
    print("---calculate scores: 4/4")
    sg0 = devprops1 - devprops2
    sg0 = pd.Series(sg0, name="sg0")
    sg0_index = sg0.index
    sg0 = scaler.fit_transform(sg0.values.reshape(-1, 1))
    sg0 = [value[0] for value in sg0]
    sg0 = pd.Series(data=sg0, index=sg0_index)
    #print("\nsg0\n", sg0.head())
    sg2 = np.log2(devprops1 + logaddition) - np.log2(devprops2 + logaddition)
    sg2 = pd.Series(sg2, name="sg2")
    sg2_index = sg2.index
    sg2 = scaler.fit_transform(sg2.values.reshape(-1, 1))
    sg2 = [value[0] for value in sg2]
    sg2 = pd.Series(data=sg2, index=sg2_index)
    #print("\nsg2\n", sg0.head())
    dg0 = (devprops1 + divaddition) / (devprops2 + divaddition)
    dg0 = pd.Series(dg0, name="dg0")
    dg0_index = dg0.index
    dg0 = scaler.fit_transform(dg0.values.reshape(-1, 1))
    dg0 = [value[0] for value in dg0]
    dg0 = pd.Series(data=dg0, index=dg0_index)
    #print("\ndg0\n", sg0.head())
    dg2 = np.log2(devprops1 + logaddition +1) / np.log2(devprops2 + logaddition +1)
    dg2 = pd.Series(dg2, name="dg2")
    dg2_index = dg2.index
    dg2 = scaler.fit_transform(dg2.values.reshape(-1, 1))
    dg2 = [value[0] for value in dg2]
    dg2 = pd.Series(data=dg2, index=dg2_index)
    #print("\ndg2\n", dg2.head())

    # Return all zeta variant scores
    return sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2, devprops1, devprops2


def get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs):
    meanrelfreqs = relativefreqs.T
    print("\nrelfreqs_df\n", meanrelfreqs.head())
    meanrelfreqs_index = meanrelfreqs.index
    meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
    meanrelfreqs = pd.Series(data=meanrelfreqs, index=meanrelfreqs_index)
    print("\nmeanrelfreqs_series\n", meanrelfreqs.head(10))
    return meanrelfreqs
    """
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r") as infile:
        meanrelfreqs = pd.read_csv(infile, sep="\t", index_col="idno").T
        print("\nrelfreqs_df\n", meanrelfreqs.head())
        meanrelfreqs_index = meanrelfreqs.index
        meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
        meanrelfreqs = pd.Series(data=meanrelfreqs, index=meanrelfreqs_index)
        print("\nmeanrelfreqs_series\n", meanrelfreqs.head(10))
        return meanrelfreqs
    """

def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs, sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2):
    #print(len(docprops1), len(docprops2), len(relfreqs1), len(relfreqs2), len(devprops1), len(devprops2), len(meanrelfreqs), len(sd0), len(sd2), len(sr0), len(sr2), len(sg0), len(sg2), len(dd0), len(dd2), len(dr0), len(dr2), len(dg0), len(dg2))
    #print(type(docprops1), type(docprops2), type(relfreqs1), type(relfreqs2), type(devprops1), type(devprops2), type(meanrelfreqs), type(sd0), type(sd2), type(sr0), type(sr2), type(sg0), type(sg2), type(dd0), type(dd2), type(dr0), type(dr2), type(dg0), type(dg2))
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
    #print(results.head())
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
    results.sort_values(by="sd0", ascending=False, inplace=True)
    print("\nresults-head\n", results.head(10), "\nresults-tail\n", results.tail(10))
    return results


def save_results(results, resultsfile):
    with open(resultsfile, "w") as outfile:
        results.to_csv(outfile, sep="\t")


# =================================
# Function: main
# =================================


def main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype, absolutefreqs, relativefreqs, binaryfreqs):
    print("--calculate")
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    #print(parameterstring)
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    #print(contraststring)
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    idlists = make_idlists(metadatafile, separator, contrast)
    #print(idlists)
    binary1, binary2, relative1, relative2, absolute1, absolute2 = filter_dtm(dtmfolder, parameterstring, idlists, absolutefreqs, relativefreqs, binaryfreqs)
    #print(binary1)
    docprops1, docprops2, relfreqs1, relfreqs2 = get_indicators(binary1, binary2, relative1, relative2)
    sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2, devprops1, devprops2 = calculate_scores(docprops1, docprops2, relfreqs1, relfreqs2, absolute1, absolute2, logaddition, segmentlength, idlists)
    meanrelfreqs = get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, devprops1, devprops2, meanrelfreqs, sd0, sd2, sr0, sr2, sg0, sg2, dd0, dd2, dr0, dr2, dg0, dg2)
    save_results(results, resultsfile)




