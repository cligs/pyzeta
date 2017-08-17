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
import treetaggerwrapper
import pygal
from pygal import style
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import random
import math
from sklearn import preprocessing as prp
from sklearn import feature_extraction as fe


# =================================
# Shared functions
# =================================


def get_filename(file):
    filename, ext = os.path.basename(file).split(".")
    print(filename)
    return filename


def read_plaintext(file):
    with open(file, "r") as infile:
        text = infile.read()
        return text


def read_csvfile(filepath):
    with open(filepath, "r", newline="\n") as csvfile:
        content = csv.reader(csvfile, delimiter='\t')
        alllines = [line for line in content]
        return alllines


def save_dataframe(allfeaturecounts, currentfile):
    with open(currentfile, "w") as outfile:
        allfeaturecounts.to_csv(outfile, sep="\t")


# =================================
# Functions: prepare
# =================================


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


def prepare(plaintextfolder, language, taggedfolder):
    print("--prepare")
    if not os.path.exists(taggedfolder):
        os.makedirs(taggedfolder)
    for file in glob.glob(plaintextfolder + "*.txt"):
        filename = get_filename(file)
        text = read_plaintext(file)
        tagged = run_treetagger(text, language)
        save_tagged(taggedfolder, filename, tagged)
    print("Done.")


# =================================
# Functions: Zeta
# =================================


def make_filelist(metadatafile, contrast):
    """
    Based on the metadata, create two lists of files, each from one group.
    The category to check and the two labels are found in Contrast.
    """
    print("--make_filelist")
    with open(metadatafile, "r") as infile:
        metadata = pd.DataFrame.from_csv(infile, sep=";")
        # print(metadata.head())
        if contrast[0] != "random": 
            onemetadata = metadata[metadata[contrast[0]].isin([contrast[1]])]
            twometadata = metadata[metadata[contrast[0]].isin([contrast[2]])]
            onelist = list(onemetadata.loc[:, "idno"])
            twolist = list(twometadata.loc[:, "idno"])
        elif contrast[0] == "random":
            idnolist = list(metadata.loc[:, "idno"])
            newidnolist = random.sample(idnolist, len(idnolist))
            #onelist = newidnolist[:int(len(idnolist)/2)]
            #twolist = newidnolist[int(len(idnolist)/2):]
            onelist = newidnolist[:189]
            twolist = newidnolist[189:339]
        #print(onelist, twolist)
        print("----number of texts: " + str(len(onelist)) + " and " + str(len(twolist)))
        return onelist, twolist


def read_stoplistfile(stoplistfile):
    print("--read_stoplistfile")
    with open(stoplistfile, "r") as infile:
        stoplist = infile.read()
        stoplist = list(re.split("\n", stoplist))
        # print(stoplist)
        return stoplist


def select_features(segment, pos, forms, stoplist):
    """
    Selects the desired features (words, lemmas or pos) from the lists of texts.
    Turns the complete list into a set, then turns into a string for better saving.
    TODO: Add a replacement feature for words like "j'" or "-ils"
    """
    if pos != "all":
        if forms == "words":
            features = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist and pos in line[1]]
        if forms == "lemmata":
            features = [line[2].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist and pos in line[1]]
        if forms == "pos":
            features = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist and pos in line[1]]
    elif pos == "all":
        if forms == "words":
            features = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist]
        if forms == "lemmata":
            features = [line[2].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist]
        if forms == "pos":
            features = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 2 and line[0] not in stoplist]
    else:
        features = []
    setfeatures = list(set(features))
    return setfeatures


def save_segment(features, segsfolder, segmentid):
    segmentfile = segsfolder + segmentid + ".txt"
    featuresjoined = " ".join(features)
    with open(segmentfile, "w") as outfile:
        outfile.write(featuresjoined)


def count_features(segment, segmentid):
    featurecount = Counter(segment)
    featurecount = dict(featurecount)
    featurecount = pd.Series(featurecount, name=segmentid)
    # print(featurecount)
    return featurecount


def make_segments(taggedfolder, currentlist, seglength,
                  segsfolder, pos, forms, stoplist, currentfile):
    """
    Calls a function to load each complete tagged text.
    Splits the whole text document into segments of fixed length; discards the rest.
    Calls a function to select the desired features from each segment.
    Calls a function to save each segment of selected features to disc.
    """
    print("--make_segments")
    allfeaturecounts = []
    for filename in currentlist:
        filepath = taggedfolder + filename + ".csv"
        if os.path.isfile(filepath):
            alllines = read_csvfile(filepath)
            numsegs = int(len(alllines) / seglength)
            for i in range(0, numsegs):
                segmentid = filename + "-" + "{:04d}".format(i)
                segment = alllines[i * seglength:(i + 1) * seglength]
                features = select_features(segment, pos, forms, stoplist)
                save_segment(features, segsfolder, segmentid)
                featurecount = count_features(features, segmentid)
                allfeaturecounts.append(featurecount)
    allfeaturecounts = pd.concat(allfeaturecounts, axis=1)
    allfeaturecounts = allfeaturecounts.fillna(0).astype(int)
    save_dataframe(allfeaturecounts, currentfile)
    # print(allfeaturecounts)
    return allfeaturecounts
  

def calculate_zetas(allfeaturecountsone, allfeaturecountstwo):
    """
    Perform the Zeta score calculation.
    Zeta = proportion of segments containing the type in group one minus proportion in group two
    """
    print("--calculate_zetas")
    # Calculate the proportions by dividing the row-wise sums by the number of segments
    allfeaturecountsone["docpropone"] = np.divide(np.sum(allfeaturecountsone, axis=1), len(allfeaturecountsone.columns))
    allfeaturecountstwo["docproptwo"] = np.divide(np.sum(allfeaturecountstwo, axis=1), len(allfeaturecountstwo.columns))
    zetascoredata = pd.concat([allfeaturecountsone.loc[:, "docpropone"], allfeaturecountstwo.loc[:, "docproptwo"]],
                              axis=1, join="outer")
    zetascoredata = zetascoredata.fillna(0)
    # The next line contains the actual zeta score calculation
    zetascoredata["zetascores"] = zetascoredata.loc[:, "docpropone"] - zetascoredata.loc[:, "docproptwo"] 

    # Adjusted zeta    
    #zetascoredata = zetascoredata.sort_values("zetascores", ascending=False)
    #zetascoredata["adjustedzeta"] = zetascoredata.loc[:,"zetascores"] * ((zetascoredata.loc[:, "docpropone"] + zetascoredata.loc[:, "docproptwo"])/2)
    #zetascoredata["adjustedzeta"] = np.log(zetascoredata.loc[:,"docpropone"]+0.3) - np.log(zetascoredata.loc[:, "docproptwo"]+0.3)
    
    # Scale adjusted zeta scores to range of original zeta for comparability
    #high = max(zetascoredata.loc[:,"zetascores"])
    #low = min(zetascoredata.loc[:,"zetascores"])
    #scaler = prp.MinMaxScaler(feature_range=(low,high))
    #scaler.fit(zetascoredata.loc[:,"adjustedzeta"])
    #zetascoredata["scaledadjustedzeta"] = scaler.transform(zetascoredata.loc[:,"adjustedzeta"])
    #zetascoredata.drop("adjustedzeta")
    
    
    #zetascoredata["difference"] = zetascoredata.loc[:,"scaledadjustedzeta"] - zetascoredata.loc[:,"zetascores"] 

    # print(zetascoredata.head(5))
    # print(zetascoredata.tail(5))
    return zetascoredata


def zeta(taggedfolder, metadatafile, contrast,
         datafolder, resultsfolder, seglength,
         pos, forms, stoplistfile, random):
    """
    Main coordinating function for "pyzeta.zeta"
    Python implementation of Craig's Zeta. 
    Status: proof-of-concept quality.
    """

    ## 1. Normal procedure 

    # Generate necessary file and folder names
    contraststring = contrast[0] + "-" + contrast[1] + "-" + contrast[2]
    parameterstring = str(seglength) + "-" + forms + "-" + str(pos)
    segsfolder = datafolder + contraststring + "_" + parameterstring + "/"
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    onefile = datafolder + "features_" + contrast[1] + "_" + parameterstring + ".csv"
    twofile = datafolder + "features_" + contrast[2] + "_" + parameterstring + ".csv"

    # Create necessary folders
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)
    if not os.path.exists(segsfolder):
        os.makedirs(segsfolder)
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)

    # Generate list of files for the two groups and get stoplist
    onelist, twolist = make_filelist(metadatafile, contrast)
    stoplist = read_stoplistfile(stoplistfile)

    # Create segments with selected types and turn into count matrix
    allfeaturecountsone = make_segments(taggedfolder, onelist, seglength, segsfolder, pos, forms, stoplist, onefile)
    allfeaturecountstwo = make_segments(taggedfolder, twolist, seglength, segsfolder, pos, forms, stoplist, twofile)

    # Perform the actual Zeta score calculation
    zetascoredata = calculate_zetas(allfeaturecountsone, allfeaturecountstwo)
    save_dataframe(zetascoredata, zetascorefile)


    ## 2. Random procedure for comparison

    if random[0] == "yes":

        # Creating the random data
        allrandomzeta = pd.DataFrame()
        counter = 0
        while counter < random[1]: 
            # Generate necessary file and folder names
            contrast = ["random", "one", "two"]
            contraststring = contrast[0] + "-" + contrast[1] + "-" + contrast[2]
            parameterstring = str(seglength) + "-" + forms + "-" + str(pos)
            segsfolder = datafolder + contraststring + "_" + parameterstring + "/"
            zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
            onefile = datafolder + "features_" + contrast[0] + "-" + contrast[1] + "_" + parameterstring + ".csv"
            twofile = datafolder + "features_" + contrast[0] + "-" + contrast[2] + "_" + parameterstring + ".csv"

            # Generate list of files for the two groups and get stoplist
            onelist, twolist = make_filelist(metadatafile, contrast)
            stoplist = read_stoplistfile(stoplistfile)

            # Create segments with selected types and turn into count matrix
            allfeaturecountsone = make_segments(taggedfolder, onelist, seglength, segsfolder, pos, forms, stoplist, onefile)
            allfeaturecountstwo = make_segments(taggedfolder, twolist, seglength, segsfolder, pos, forms, stoplist, twofile)

            # Perform the actual Zeta score calculation
            randomzetascoredata = calculate_zetas(allfeaturecountsone, allfeaturecountstwo)
            justrandomzeta = randomzetascoredata.drop(["docpropone", "docproptwo"], axis=1)
            justrandomzeta = pd.Series(justrandomzeta.loc[:,"zetascores"])           
            #print(justrandomzeta[0:5])

            # Merge one random set of scores into the others
            allrandomzeta[str(counter)] = justrandomzeta.values
            counter +=1

        # Get some distribution indicators for each value
        mins = np.min(allrandomzeta, axis=1)
        maxs = np.max(allrandomzeta, axis=1)
        means = np.mean(allrandomzeta, axis=1)
        medians = np.median(allrandomzeta, axis=1)
        stds = np.std(allrandomzeta, axis=1)
        sems = stats.sem(allrandomzeta, axis=1)
        cin950mins, cin950maxs = stats.norm.interval(0.950, loc=means, scale=stds)
        #cin990mins, cin990maxs = stats.norm.interval(0.990, loc=means, scale=stds)
        cin999mins, cin999maxs = stats.norm.interval(0.999, loc=means, scale=stds)
        cin9999mins, cin9999maxs = stats.norm.interval(0.9999, loc=means, scale=stds)

        # Add those indicators to the dataframe
        allrandomzeta["min"] = mins
        allrandomzeta["max"] = maxs
        allrandomzeta["mean"] = means
        allrandomzeta["median"] = medians
        allrandomzeta["std"] = stds
        allrandomzeta["sem"] = sems
        allrandomzeta["cin950min"] = cin950mins
        allrandomzeta["cin950max"] = cin950maxs
        #allrandomzeta["cin990min"] = cin990mins
        #allrandomzeta["cin990max"] = cin990maxs
        allrandomzeta["cin999min"] = cin999mins
        allrandomzeta["cin999max"] = cin999maxs
        allrandomzeta["cin9999min"] = cin9999mins
        allrandomzeta["cin9999max"] = cin9999maxs

        # Get the list of features
        zetascoredata["features"] = zetascoredata.index

        # Add the data from the "real" run to the table
        allrandomzeta["features"] = pd.Series(zetascoredata.loc[:,"features"].values)
        allrandomzeta["docpropone"] = pd.Series(zetascoredata.loc[:,"docpropone"].values)
        allrandomzeta["docproptwo"] = pd.Series(zetascoredata.loc[:,"docproptwo"].values)
        allrandomzeta["zetascores"] = pd.Series(zetascoredata.loc[:,"zetascores"].values)
        
        #allrandomzeta["diff950"] = [np.abs(allrandomzeta.loc[:,"zetascores"]) - np.abs(allrandomzeta.loc[:,"cin950max"]) if x > 0 else np.abs(allrandomzeta.loc[:,"zetascores"]) - np.abs(allrandomzeta.loc[:,"cin950min"]) for x in allrandomzeta.loc[:,"zetascores"]]
        print(allrandomzeta.head())
        #allrandomzeta["sig950"] = [1 if x > 0 else 0 for x in allrandomzeta.loc[:,"diff950"]]
        
 
        # Save everything
        print(allrandomzeta.head())
        save_dataframe(allrandomzeta, zetascorefile)




    


# =================================
# Functions: plot zetadata
# =================================

zeta_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family="FreeSans",
    title_font_size = 18,
    legend_font_size = 14,
    label_font_size = 12,
    major_label_font_size = 12,
    value_font_size = 12,
    major_value_font_size = 12,
    tooltip_font_size = 12,
    opacity_hover=0.2)


def calculate_confint(zetadata):
    """
    Calculate the confidence interval for the zeta score distribution.
    Status: Not useful because a huge part of the scores are beyond the confint.
    """
    print("----calculate_confint")
    zetascores = list(zetadata.loc[:, "zetascores"])
    numvals = len(zetascores)
    mean = np.mean(zetascores)
    sem = stats.sem(zetascores)
    coveredint = 0.9995  # the area of the distribution to be excluded
    confint = stats.norm.interval(coveredint,
                                   loc=mean,
                                   scale=sem)
    # print(confint)
    return confint


def get_zetadata(zetascorefile, numwords):
    print("----get_zetadata")
    with open(zetascorefile, "r") as infile:
        zetadata = pd.DataFrame.from_csv(infile, sep="\t")
        # print(zetadata.head())
        zetadata.drop(["docpropone", "docproptwo"], axis=1, inplace=True)
        zetadata.sort_values("zetascores", ascending=False, inplace=True)
        confint = calculate_confint(zetadata)
        zetadatahead = zetadata.head(numwords)
        zetadatatail = zetadata.tail(numwords)
        zetadata = zetadatahead.append(zetadatatail)
        zetadata = zetadata.reset_index(drop=False)
        # print(zetadata.head())
        return zetadata, confint


def plot_zetadata(zetadata, contrast, contraststring, zetaplotfile, numwords):
    print("----plot_zetadata")
    plot = pygal.HorizontalBar(style = zeta_style,
                               print_values = False,
                               print_labels = True,
                               show_legend = False,
                               range = (-1, 1),
                               title = ("Kontrastive Analyse mit Zeta\n (" +
                                        str(contrast[0]) + ": " + str(contrast[2]) + " vs. " + str(contrast[1]) + ")"),
                               x_title = "Zeta-Score",
                               y_title = str(numwords) + " Worte pro Partition"
                               )
    for i in range(len(zetadata)):
        if zetadata.iloc[i, 1] > 0.8:
            color = "#00cc00"
        if zetadata.iloc[i, 1] > 0.7:
            color = "#14b814"
        if zetadata.iloc[i, 1] > 0.6:
            color = "#29a329"
        elif zetadata.iloc[i, 1] > 0.5:
            color = "#3d8f3d"
        elif zetadata.iloc[i, 1] > 0.3:
            color = "#4d804d"
        elif zetadata.iloc[i, 1] < -0.8:
            color = "#0066ff"
        elif zetadata.iloc[i, 1] < -0.7:
            color = "#196be6"
        elif zetadata.iloc[i, 1] < -0.6:
            color = "#3370cc"
        elif zetadata.iloc[i, 1] < -0.5:
            color = "#4d75b3"
        elif zetadata.iloc[i, 1] < -0.3:
            color = "#60799f"
        else:
            color = "#585858"
        plot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 1], "label": zetadata.iloc[i, 0], "color": color}])
    plot.render_to_file(zetaplotfile)


def plot_zetascores(numfeatures, contrast, contraststring, parameterstring, resultsfolder):
    print("--plot_zetascores")
    # Define some filenames
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    zetaplotfile = resultsfolder + "scoresplot_" + contraststring + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    # Get the data and plot it
    zetadata, confint = get_zetadata(zetascorefile, numfeatures)
    plot_zetadata(zetadata, contrast, contraststring, zetaplotfile, numfeatures)




# ==============================================
# Plot real and random zeta data together
# ==============================================


def get_realrandomdata(realrandomdatafile, numfeatures):
    print("----get_realrandomdata")
    with open(realrandomdatafile, "r") as infile:
        realrandomdata = pd.DataFrame.from_csv(infile, sep="\t")
        realrandomdatahead = realrandomdata.head(numfeatures)
        realrandomdatatail = realrandomdata.tail(numfeatures)
        realrandomdata = realrandomdatahead.append(realrandomdatatail)
        realrandomdata = realrandomdata.reset_index(drop=False)
        print(realrandomdata)
        return realrandomdata


def plot_rrdata(realrandomdata, realrandomplotfile, contrast, contraststring, numfeatures):
    print("----plot_rrdata")
    plot = pygal.HorizontalBar(style = zeta_style,
                               print_values = False,
                               print_labels = True,
                               show_legend = False,
                               range = (-1, 1),
                               title = ("Kontrastive Analyse mit Zeta\n (" +
                                        str(contrast[0]) + ": " + str(contrast[2]) + " vs. " + str(contrast[1]) + ")"),
                               x_title = "Zeta-Score",
                               y_title = str(numfeatures) + " Worte pro Partition"
                               )
    for i in range(0, 2000):
        feature = realrandomdata.loc[i, "features"]
        label = realrandomdata.loc[i, "features"]
        realzeta = realrandomdata.loc[i, "zetascores"]
        mean = realrandomdata.loc[i, "mean"]
        lowvalue = realrandomdata.loc[i, "cin999min"] # equals 95% confidence interval
        highvalue = realrandomdata.loc[i, "cin999max"] # equals 95% confidence interval
        if realzeta > highvalue:
            color = "green"
        else:
            color = "#585858"
        #ppppppppot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 1], "label": zetadata.iloc[i, 0], "color": color}])
        #plot.add(feature, [{"value" : realzeta, "label" : str(label), "ci" : {"low" : lowvalue, "high" : highvalue}, "color": color}])
        plot.add(feature, [{"value" : realzeta, "ci" : {"low" : lowvalue, "high" : highvalue}, "color": color}])
    for i in range(2000, 4000):
        feature = realrandomdata.loc[i, "features"]
        label = realrandomdata.loc[i, "features"]
        realzeta = realrandomdata.loc[i, "zetascores"]
        mean = realrandomdata.loc[i, "mean"]
        lowvalue = realrandomdata.loc[i, "cin950max"] # equals 95% confidence interval
        highvalue = realrandomdata.loc[i, "cin950min"] # equals 95% confidence interval
        if realzeta < highvalue:
            color = "red"
        else:
            color = "#585858"
        #ppppppppot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 1], "label": zetadata.iloc[i, 0], "color": color}])
        #plot.add(feature, [{"value" : realzeta, "label" : str(label), "ci" : {"low" : lowvalue, "high" : highvalue}, "color": color}])
        plot.add(feature, [{"value" : realzeta, "ci" : {"low" : lowvalue, "high" : highvalue}, "color": color}])

    plot.render_to_file(realrandomplotfile)
    



def plot_realrandom(numfeatures, contrast, contraststring, parameterstring, resultsfolder):
    print("--plot_realrandom")
    contrast = ["random", "one", "two"]
    contraststring = contrast[0] + "-" + contrast[1] + "-" + contrast[2]
    realrandomdatafile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    realrandomplotfile = resultsfolder + "scoresplot_" + contraststring + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    realrandomdata = get_realrandomdata(realrandomdatafile, numfeatures)
    plot_rrdata(realrandomdata, realrandomplotfile, contrast, contraststring, numfeatures)
    















# ==============================================
# Scatterplot of types
# ==============================================


def get_scores(zetascorefile, numfeatures):
    print("----get_scores")
    with open(zetascorefile, "r") as infile:
        zetascores = pd.DataFrame.from_csv(infile, sep="\t")
        positivescores = zetascores.head(numfeatures)
        negativescores = zetascores.tail(numfeatures)
        scores = pd.concat([positivescores, negativescores])
        # print(scores.head())
        return scores


def make_data(scores):
    print("----make_data")
    thetypes = list(scores.index)
    propsone = list(scores.loc[:, "docpropone"])
    propstwo = list(scores.loc[:, "docproptwo"])
    zetas = list(scores.loc[:, "zetascores"])
    return thetypes, propsone, propstwo, zetas


def make_typesplot(types, propsone, propstwo, zetas, numfeatures, cutoff, contrast, typescatterfile):
    print("----make_typesplot")
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range=(0, 1),
                    show_y_guides=True,
                    show_x_guides=True,
                    title="Document proportions and Zeta",
                    x_title="document proportions in " + str(contrast[1]),
                    y_title="document proportions in " + str(contrast[2]))
    for i in range(0, numfeatures * 2):
        if zetas[i] > cutoff:
            color = "green"
            size = 4
        elif zetas[i] < -cutoff:
            color = "blue"
            size = 4
        else:
            color = "grey"
            size = 3
        plot.add(str(types[i]), [
            {"value": (propsone[i], propstwo[i]), "label": "zeta " + str(zetas[i]), "color": color,
             "node": {"r": size}}])
        plot.add("orientation", [(0, 0.3), (0.7, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0, 0.6), (0.4, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0.3, 0), (1, 0.7)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0.6, 0), (1, 0.4)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0, 0), (1, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
    plot.render_to_file(typescatterfile)


def plot_types(numfeatures, cutoff, contrast, contraststring, parameterstring, resultsfolder):
    """
    Function to make a scatterplot with the type proprtion data.
    """
    print("--plot_types")
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    typescatterfile = (resultsfolder + "typescatter_" + contraststring + "_"
                       + parameterstring + "-" + str(numfeatures) + "-" + str(cutoff) + ".svg")
    scores = get_scores(zetascorefile, numfeatures)
    thetypes, propsone, propstwo, zetas = make_data(scores)
    make_typesplot(thetypes, propsone, propstwo, zetas, numfeatures, cutoff, contrast, typescatterfile)


# ==============================================
# Plot random-based Zeta distributio for a feature
# ==============================================


def get_randomdist(zetascorefile, feature): 
    print("----get_randomdist")
    with open(zetascorefile, "r") as infile:
        zetascores = pd.DataFrame.from_csv(infile, sep="\t")
        #print(zetascores.head())
        randomscores = zetascores.iloc[feature,:]
        #print(randomscores.iloc[0:20])
        return randomscores

        
def plot_randomscores(randomscores, randomdistfile): 
    print("----plot_randomscores")
    import seaborn as sns
    randomscores = list(randomscores)
    randomscores = sorted(randomscores[0:-4])
    normality = stats.mstats.normaltest(randomscores)
    print(normality[1])
    histogram = sns.distplot(randomscores)
    histogram = histogram.get_figure()
    histogram.savefig(randomdistfile, dpi=300)
    ax = plt.axes()
    ax.set_title(str(normality[1]))
    plt.close(histogram)
        

def plot_randomdist(feature, contrast, contraststring, parameterstring, resultsfolder): 
    """
    Plot the distribution of Zeta scores for a given rank/feature
    """
    contraststring = "random-one-two"
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    randomdistfile = resultsfolder + "randomdist_" + contraststring + "_" + parameterstring + "-ft" + str(feature) + ".png"
    randomscores = get_randomdist(zetascorefile, feature)
    plot_randomscores(randomscores, randomdistfile)
    
    
    
    
    
    
    
# ==============================================
# Zeta-based Scatterplot of works
# ==============================================


def get_zetawords(zetascorefile, features): 
    """
    Get the list of n features with most extreme Zeta values.
    This will be the features on which the following calculations are based.
    """
    with open(zetascorefile, "r") as infile: 
        zetascores = pd.DataFrame.from_csv(infile, sep="\t")
        allzetawords = zetascores.loc[:,"zetascores"].index
        poszetawords = list(allzetawords[:features]) 
        negzetawords = list(allzetawords[-features:])
        return poszetawords, negzetawords

def get_lemmatext(file): 
    """
    Read the tagged files and extract the lemmata.
    """
    idno,ext = os.path.basename(file).split(".")
    with open(file, "r") as infile: 
        tagged = pd.DataFrame.from_csv(infile, sep="\t")
        lemmatext = list(tagged.iloc[:,1].values)
        lemmatext = [str(word).lower() for word in lemmatext]
        #print(lemmatext[0:5])
        return idno, lemmatext
                
    
def get_zetaproportions(idno, lemmatext, poszetawords, negzetawords): 
    """
    For each document, calculate the proportion of markers and anti-markers.
    Return a list of document identifiers with these two proportions.
    """
    poscount = 0
    negcount = 0
    lengthoftext = len(lemmatext)
    for lemma in lemmatext: 
        if lemma in poszetawords:
            poscount +=1
        elif lemma in negzetawords: 
            negcount +=1
    posprop = poscount/len(lemmatext)
    negprop = negcount/len(lemmatext)
    posprop = float("{:03.2f}".format(posprop*100))
    negprop = float("{:03.2f}".format(negprop*100))
    #print(idno, posprop, negprop)
    return posprop, negprop

    
def read_metadatafile(metadatafile): 
    """
    Reads the metadatafile. Output is a DataFrame.
    """
    with open(metadatafile, "r") as infile: 
        metadata = pd.DataFrame.from_csv(infile, sep=";")
        #print(metadata.head())
        return metadata
  
        
def get_category(idno, metadata): 
    """
    For each identifier, get the category label from the metadatafile.
    This serves to color the points in the plot by category.
    """
    category = metadata.loc[idno, "subgenre"]
    author = metadata.loc[idno, "author-name"]
    form = metadata.loc[idno, "form"]
    year = metadata.loc[idno, "year"]
    return category, author, form, year


def plot_worksbyzetaprop(worksbyzetaprop, contrast, worksbyzetaplotfile, features): 
    """
    Create a scatterplot in which each work is placed. 
    Position depends on proportion of lemmata from top-/bottom-zeta features. 
    Works from the same category should group together. 
    """    
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range = (0,8),
                    show_y_guides=True,
                    show_x_guides=True,
                    title="Works by percentage of "+ str(features)+" positive and negative Zeta words",
                    x_title="percentage of tragedy words",
                    y_title="percentage of comedy words")
    for item in worksbyzetaprop: 
        if item[3] == "tragedie": 
            color = "navy"
            size = 4
        elif item[3] == "comedie": 
            color = "darkred"
            size = 4
        else: 
            color = "grey"
            size = 2
        label = str(item[3]) +" ("+ str(item[4]) +", "+ str(item[6]) +", "+ str(item[5]) + ", " + str(item[0]) +")"
        plot.add(label, [{"value" : (item[1], item[2]), "color" : color, "node": {"r": size}}])
    plot.add("orientation", [(0, 0), (8, 8)], stroke=True, show_dots=False,
             stroke_style={'width': 0.5, 'dasharray': '2, 6'})
    plot.render_to_file(worksbyzetaplotfile)
     

def works_by_zeta(features, taggedfolder, metadatafile, contrast, contraststring, parameterstring, resultsfolder): 
    print("--works_by_zeta", features)
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"    
    worksbyzetaplotfile = resultsfolder + "worksbyzeta_" + contraststring + "_" + parameterstring + "_" + str(features) + ".svg"
    metadata = read_metadatafile(metadatafile)
    worksbyzetaprop = []
    poszetawords, negzetawords = get_zetawords(zetascorefile, features)
    for file in glob.glob(taggedfolder+"*.csv"): 
        idno, lemmatext = get_lemmatext(file)
        posprop, negprop = get_zetaproportions(idno, lemmatext, poszetawords, negzetawords)
        category, author, year, form = get_category(idno, metadata)
        worksbyzetaprop.append([idno, posprop, negprop, category, author, year, form])
    #print(worksbyzetaprop[0])
    plot_worksbyzetaprop(worksbyzetaprop, contrast, worksbyzetaplotfile, features)





    
# ==============================================
# Threeway comparison
# ==============================================

threeway_style = pygal.style.Style(
    background = 'white',
    plot_background = 'white',
    font_family = "FreeSans",
    guide_stroke_dasharray = (6,6),
    major_guide_stroke_dasharray = (1,1),
    title_font_size = 18,
    legend_font_size = 14,
    label_font_size = 12,
    major_label_font_size = 12,
    value_font_size = 12,
    major_value_font_size = 12,
    tooltip_font_size = 12,
    opacity_hover = 0.2,
    colors = ("firebrick", "mediumblue", "green"))


def get_distfeatures(zetascorefile, numfeatures):
    print("----get_distfeatures")
    with open(zetascorefile, "r") as infile:
        allzetascores = pd.DataFrame.from_csv(infile, sep="\t")
        allzetascores["type"] = allzetascores.index
        distscoreshead = allzetascores.head(numfeatures)
        distscorestail = allzetascores.tail(numfeatures)
        distscoresall = distscoreshead.append(distscorestail)
        distfeatures = list(distscoresall.loc[:, "type"])
        # print("distfeatures", distfeatures)
        return distfeatures


def select_distrawcounts(featuresfile, distfeatures):
    print("----select_distrawcounts")
    with open(featuresfile, "r") as infile:
        allcounts = pd.DataFrame.from_csv(infile, sep="\t")
        allcounts["type"] = allcounts.index
        # print(allcounts.head())
        distrawcounts = allcounts[allcounts["type"].isin(distfeatures)]
        distrawcounts = distrawcounts.drop("type", axis=1)
        # print(distrawcounts.head())
        return distrawcounts


def calculate_distprops(distrawcounts, group):
    print("----calculate_distprops")
    distprops = np.sum(distrawcounts, axis=1).divide(len(distrawcounts.columns), axis=0)
    distprops = pd.Series(distprops, name=group)
    # print(distprops)
    return distprops


def load_dataframe(distpropsfile):
    with open(distpropsfile, "r") as infile:
        alldistprops = pd.DataFrame.from_csv(infile, sep="\t")
        # print(alldistprops.head())
        return alldistprops


def make_dotplot(alldistprops, sortby, dotplotfile):
    print("----make_dotplot")
    alldistprops = alldistprops.T
    alldistprops = alldistprops.sort_values(sortby, axis=0, ascending="False")
    alldistprops = alldistprops.T
    dotplot = pygal.Dot(style=threeway_style,
                        show_legend=False,
                        legend_at_bottom=True,
                        legend_at_bottom_columns=3,
                        show_y_guides=True,
                        show_x_guides=False,
                        x_label_rotation=60,
                        title="Vergleich dreier Gruppen\n(Anteil der Segmente pro Gruppe)",
                        x_title="Distinktive Types",
                        y_title="Textgruppen")
    distfeatures = alldistprops.columns
    dotplot.x_labels = distfeatures
    for i in range(0,3):
        grouplabel = alldistprops.index[i]
        distprops = list(alldistprops.loc[alldistprops.index[i],:])
        dotplot.add(grouplabel, distprops)
    dotplot.render_to_file(dotplotfile)


    
def make_lineplot(alldistprops, sortby, lineplotfile):
    print("----make_lineplot")
    if sortby == "zetascores":
        alldistprops = alldistprops.T
        alldistprops[sortby] = alldistprops.loc[:,"comedie"] - alldistprops.loc[:,"tragedie"]
        alldistprops = alldistprops.sort_values(sortby, axis=0, ascending=False)
        print(alldistprops)
        alldistprops = alldistprops.T
    else:
        alldistprops = alldistprops.T
        alldistprops = alldistprops.sort_values(sortby, axis=0, ascending="True")
        alldistprops = alldistprops.T
    lineplot = pygal.Line(style=threeway_style,
                    show_legend=True,
                    legend_at_bottom=True,
                    legend_at_bottom_columns=3,
                    show_y_guides=False,
                    show_x_guides=False,
                    x_label_rotation=60,
                    title="Vergleich dreier Gruppen",
                    x_title="Distinktive Types",
                    y_title="Anteil der Segmente",
                    interpolate='cubic')
    distfeatures = alldistprops.columns
    lineplot.x_labels = distfeatures
    for i in range(0,3):
        grouplabel = alldistprops.index[i]
        distprops = list(alldistprops.loc[alldistprops.index[i],:])
        lineplot.add(grouplabel, distprops)
        lineplot.render_to_file(lineplotfile)


def test_correlations(alldistprops):
    print("----test_correlations")
    columnlabels = ["groupone", "grouptwo", "correlation", "p-value"]
    allcorrinfos = []
    for i,j in [(0,1), (0,2), (1,2)]:
        comparison = [alldistprops.index[i], alldistprops.index[j]]
        correlation = stats.pearsonr(list(alldistprops.loc[alldistprops.index[i],:]),
                                     list(alldistprops.loc[alldistprops.index[j],:]))
        corrinfo = [comparison[0], comparison[1], correlation[0], correlation[1]]
        allcorrinfos.append(corrinfo)
    allcorrinfosdf = pd.DataFrame(allcorrinfos, columns=columnlabels)
    allcorrinfosdf.sort_values(by="p-value", ascending=True, inplace=True)
    # print(allcorrinfosdf)
    return allcorrinfosdf


# Coordinating function
def threeway_compare(datafolder, resultsfolder, contrast, contraststring, parameterstring,
             thirdgroup, numfeatures, sortby, mode):
    print("--threeway_compare")
    # Create necessary filenames
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    dotplotfile = resultsfolder + "dotplot_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    lineplotfile = resultsfolder + "lineplot_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    distpropsfile = datafolder + "distpropspergroup_" + contraststring + "_" + parameterstring + "-" + str(numfeatures) + ".csv"
    correlationsfile = resultsfolder + "correlations_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".csv"
    # Do the actual work
    if mode == "generate":
        # Calculate the proportions for each feature for each group as a whole
        distfeatures = get_distfeatures(zetascorefile, numfeatures)
        alldistprops = pd.DataFrame()
        for group in (contrast[1], contrast[2], thirdgroup[1]):
            featuresfile = datafolder + "features_" + group + "_" + parameterstring + ".csv"
            distrawcounts = select_distrawcounts(featuresfile, distfeatures)
            distprops = calculate_distprops(distrawcounts, group)
            alldistprops = alldistprops.append(distprops)
        save_dataframe(alldistprops, distpropsfile)
        # Visualize the data and make a correlation test
        make_dotplot(alldistprops, sortby, dotplotfile)
        make_lineplot(alldistprops, sortby, lineplotfile)
        correlationscoresdf = test_correlations(alldistprops)
        save_dataframe(correlationscoresdf, correlationsfile)
    if mode == "analyze":
        # Load data from a previous "generate" step
        alldistprops = load_dataframe(distpropsfile)
        # print(alldistprops)
        # Visualize the data and make a correlation test
        make_dotplot(alldistprops, sortby, dotplotfile)
        make_lineplot(alldistprops, sortby, lineplotfile)
        correlationscoresdf = test_correlations(alldistprops)
        save_dataframe(correlationscoresdf, correlationsfile)


# ==============================================
# Threeway clustering
# ==============================================


# ========= make_boxplots =================


def make_propspertext(distrawcounts, label):
    distrawcounts = distrawcounts.T
    segids = list(distrawcounts.index)
    distrawcounts["idnos"] = [item[0:6] for item in segids]
    # print(distrawcounts.head())
    rawcountspertext = pd.groupby(distrawcounts, "idnos")
    distpropspertext = rawcountspertext.aggregate(np.mean)
    distpropspertext["label"] = label
    # print(distpropspertext.head())
    return distpropspertext


def make_boxplots(alldistpropspertext, boxplotfile):
    for feature in alldistpropspertext.columns[:-1].values:
        currentboxplotfile = boxplotfile + "_" + feature + ".svg"
        distribution = alldistpropspertext.loc[:,[feature,"label"]]
        distributionspertext = pd.groupby(distribution, "label")
        boxplot = pygal.Box(style=threeway_style,
                            title = "Verteilung für \'" + feature + "\'",
                            x_title = "Partitionen",
                            y_title = "Verteilung der Anteile",
                            legend_at_bottom=True,
                            legend_at_bottom_columns = 3,
                            )
        for name, group in distributionspertext:
            groupprops = list(group.loc[:,feature])
            boxplot.add(name, groupprops)
        boxplot.render_to_file(currentboxplotfile)


# ========= perform_clusternalysis =================


def get_labels(alldistpropspertext):
    """
    Get the labels for each text: idno and group
    """
    print("----get_labels")
    groups = list(alldistpropspertext.iloc[:, -1])
    idnos = list(alldistpropspertext.index.values)
    labels = []
    for i in range(len(idnos)):
        label = idnos[i] + "-" + groups[i]
        labels.append(label)
    # print(labels)
    return groups, idnos, labels


def scale_features(alldistpropspertext):
    """
    4: Apply feature scaling to the data, in this case transform to z-scores.
    """
    print("----scale_features")
    alldistpropspertext = alldistpropspertext.iloc[:,0:-2]
    # print(alldistpropspertext.head())
    # Define the means and std of each word
    means = np.mean(alldistpropspertext, axis=1)
    stds = np.std(alldistpropspertext, axis=1)
    # Substract the mean and divide by the std for each word
    scaled = alldistpropspertext
    scaled = scaled.subtract(means, axis=0)
    scaled = scaled.divide(stds, axis=0)
    # print(scaled.head())
    # print(len(scaled))
    return scaled


def get_distancematrix(scaled, distmeasure):
    """
    5: Transform the term-document matrix to a distance matrix.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    """
    print("----get_distancematrix")
    distancematrix = pdist(scaled, distmeasure)
    # print(distancematrix)
    return distancematrix


def get_linkagematrix(distancematrix):
    """
    6: Transform the distance matrix to a linkage matrix.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    print("----get_linkagematrix")
    linkagematrix = linkage(distancematrix, method="ward", metric="Euclidean")
    # print(linkagematrix[0:10])
    return linkagematrix


def make_dendrogram(linkagematrix, labels, dendrogramfile):
    """
    7: Visualize the linkage matrix as a dendrogam.
    """
    print("----make_dendrogram")
    plt.figure(figsize=(10,20))
    plt.title("Dendrogramm", fontsize=12)
    matplotlib.rcParams['lines.linewidth'] = 0.6
    dendrogram(linkagematrix,
               orientation="left",
               labels = labels,
               leaf_font_size = 4,
               )
    plt.savefig(dendrogramfile, dpi=600, figsize=(6,12), bbox_inches="tight")
    plt.close()


def perform_clusteranalysis(alldistpropspertext, distmeasure, dendrogramfile):
    print("----perform_clusteranalysis")
    groups, idnos, labels = get_labels(alldistpropspertext)
    scaled = scale_features(alldistpropspertext)
    distancematrix = get_distancematrix(scaled, distmeasure)
    linkagematrix = get_linkagematrix(distancematrix)
    make_dendrogram(linkagematrix, labels, dendrogramfile)


# ========= perform_pca =================


def apply_pca(alldistpropspertext):
    alldistpropspertext = alldistpropspertext.iloc[:, 0:-2]
    pca = PCA(n_components=5, whiten=False)
    pca.fit(alldistpropspertext)
    variance = pca.explained_variance_ratio_
    transformed = pca.transform(alldistpropspertext)
    # print(transformed)
    # print(variance)
    return transformed, variance


def make_2dscatterplot(transformed, variance, groups, idnos, pcafile):
    print("----make_2dscatterplot")
    components = [0, 2]
    plot = pygal.XY(style=zeta_style,
                    stroke=False,
                    show_legend=False,
                    show_y_guides=True,
                    show_x_guides=False,
                    title="PCA mit distinktiven Features",
                    x_title="PC" + str(components[0] + 1) + " (" + "{:03.2f}".format(variance[components[0]]*100) + "%)",
                    y_title="PC" + str(components[1] + 1) + " (" + "{:03.2f}".format(variance[components[1]]*100) + "%)",
                    )
    for i in range(0, len(idnos)):
        label = groups[i]
        point = (transformed[i][components[0]], transformed[i][components[1]])
        idno = idnos[i]
        if label == "comedie":
            mycolor = "firebrick"
        elif label == "tragedie":
            mycolor = "mediumblue"
        elif label == "tragicomedie":
            mycolor = "green"
        else:
            mylabel = "ERROR"
            mycolor = "grey"
        plot.add(idno, [{"value": point, "label": label, "color": mycolor}], dots_size=4)
    plot.render_to_file(pcafile)


def make_pcboxplot(transformed, variance, groups, idnos, pcboxplotfile):
    print("----make_pcboxplot")
    values = []
    for i in range(0,391):
        value = transformed[i][0]
        values.append(value)
    data = pd.DataFrame({"group" : groups, "value" : values, "idnos" : idnos})
    # print(data.head())
    data = data.groupby("group")
    boxplot = pygal.Box(style=threeway_style,
                        title="Verteilung der Texte",
                        # x_title="X",
                        y_title="PC 1",
                        legend_at_bottom=True,
                        print_labels = True,
                        legend_at_bottom_columns=3)
    for name, group in data:
        groupscores = list(group.loc[:, "value"])
        boxplot.add(name, groupscores)
    boxplot.render_to_file(pcboxplotfile)
    return data


def test_mannwhitney(data):
    """
    # Performs the Mann-Whitney-U-test for two independent samples that may not be normally distributed.
    # See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    # Status: Ok, but not entirely confident all is correct.
    """
    labels = ["comedie", "tragedie", "tragicomedie"]
    allgroupscores = []
    for name, group in data:
        groupscores = list(group.loc[:, "value"])
        allgroupscores.append(groupscores)
    combinations = [[0,1], [0,2], [1,2]]
    for combination in combinations:
        mannwhitney =  stats.mannwhitneyu(allgroupscores[combination[0]], allgroupscores[combination[1]], alternative="two-sided")
        statistics = [labels[combination[0]], labels[combination[1]], mannwhitney[0], mannwhitney[1]]
        print(statistics)


def perform_pca(alldistpropspertext, pcafile, pcboxplotfile):
    print("----perform_pca")
    groups, idnos, labels = get_labels(alldistpropspertext)
    transformed, variance = apply_pca(alldistpropspertext)
    make_2dscatterplot(transformed, variance, groups, idnos, pcafile)
    data = make_pcboxplot(transformed, variance, groups, idnos, pcboxplotfile)
    test_mannwhitney(data)


# ========= threeway_clustering: coordinating function =================
def threeway_clustering(datafolder, resultsfolder, contrast, contraststring, parameterstring,
                        thirdgroup, numfeatures, distmeasure, mode):
    print("--threeway_clustering")
    # Define necessary filenames
    zetascorefile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".csv"
    distpropspertextfile = datafolder + "distpropspertext_" + contraststring + "_" + parameterstring + "-" + str(numfeatures) + ".csv"
    boxplotfile = resultsfolder + "boxplot_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures)
    dendrogramfile = resultsfolder + "dendrogram_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".png"
    pcafile = resultsfolder + "pca_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    pcboxplotfile = resultsfolder + "pcboxplot_" + contraststring + "-" + thirdgroup[1] + "_" + parameterstring + "-" + str(numfeatures) + ".svg"
    # Get the necessary data
    distfeatures = get_distfeatures(zetascorefile, numfeatures)
    alldistpropspertext = pd.DataFrame()
    if mode == "generate":
        # Calculate the proportions for each feature for each text in the three groups
        for group in (contrast[1], contrast[2], thirdgroup[1]):
            label = str(group)
            featuresfile = datafolder + "features_" + group + "_" + parameterstring + ".csv"
            distrawcounts = select_distrawcounts(featuresfile, distfeatures)
            distpropspertext = make_propspertext(distrawcounts, label)
            alldistpropspertext = alldistpropspertext.append(distpropspertext)
            save_dataframe(alldistpropspertext, distpropspertextfile)
        # make_boxplots(alldistpropspertext, boxplotfile)
        perform_clusteranalysis(alldistpropspertext, distmeasure, dendrogramfile)
    if mode == "analyze":
        # Load data from a previous "generate" step
        alldistpropspertext = load_dataframe(distpropspertextfile)
        # make_boxplots(alldistpropspertext, boxplotfile)
        # perform_clusteranalysis(alldistpropspertext, distmeasure, dendrogramfile)
        perform_pca(alldistpropspertext, pcafile, pcboxplotfile)


        
        
        
        
# =====================================
# ration of relative frequecies 
# =====================================        


def make_lemmacounts(onelist, twolist, taggedfolder):
    onelemmatext = []
    twolemmatext = []
    for idno in onelist: 
        file = taggedfolder + idno + ".csv"
        idno, lemmatext = get_lemmatext(file)
        onelemmatext.extend(lemmatext)
    for idno in twolist: 
        file = taggedfolder + idno + ".csv"
        idno, lemmatext = get_lemmatext(file)
        twolemmatext.extend(lemmatext)
    onelemmatext = [lemma for lemma in onelemmatext if lemma not in [".", ",", ";", "?", "!"]]
    twolemmatext = [lemma for lemma in twolemmatext if lemma not in [".", ",", ";", "?", "!"]]
    onelength = len(onelemmatext)    
    twolength = len(twolemmatext)    
    print("----number of lemmata:", onelength, "and", twolength)
    onelemmacounts = pd.Series(dict(Counter(onelemmatext)), name="tragedy")
    twolemmacounts = pd.Series(dict(Counter(twolemmatext)), name="comedy")
    lemmacounts = pd.concat([onelemmacounts, twolemmacounts], axis=1, join="inner")
    return lemmacounts, onelength, twolength


def get_rrfs(lemmacounts, onelength, twolength): 
    lemmacounts.loc[:,"tragedy"] = lemmacounts.loc[:,"tragedy"] / onelength * 100
    lemmacounts.loc[:,"comedy"] = lemmacounts.loc[:,"comedy"] / twolength * 100
    lemmacounts["rrf-tragedy"] = lemmacounts.loc[:,"tragedy"] / lemmacounts.loc[:,"comedy"]
    lemmacounts["rrf-comedy"] = lemmacounts.loc[:,"comedy"] / lemmacounts.loc[:,"tragedy"]
    #rrfs = lemmacounts.sort_values(by="tragedy", ascending=False)
    #print(rrfs.head(10))
    #rrfs = lemmacounts.sort_values(by="comedy", ascending=False)
    #print(rrfs.head(10))
    rrfs = lemmacounts.sort_values(by="rrf-tragedy", ascending=False)
    print(rrfs.head(10))
    rrfs = lemmacounts.sort_values(by="rrf-comedy", ascending=False)
    print(rrfs.head(10))
    return rrfs

    
def rrf(taggedfolder, metadatafile, contrast, datafolder, resultsfolder, seglength, pos, forms, stoplistfile, random):
    onelist, twolist = make_filelist(metadatafile, contrast)
    lemmacounts, onelength, twolength = make_lemmacounts(onelist, twolist, taggedfolder)
    rrfs = get_rrfs(lemmacounts, onelength, twolength)
    with open(resultsfolder + "rrfs_tragedy-comedy.csv", "w") as outfile: 
        rrfs.to_csv(outfile, sep=";")
    
    
    
# =============================================
# classify
# =============================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: classify_bytopics.py
# Author: #cf (2016)

import re
import os
import glob
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
import pygal



################################
# Functions
################################


def define_classifier(classifiertype): 
    """
    Select and define the type of classifier. 
    Called by "classify_data"
    SVM: http://scikit-learn.org/stable/modules/svm.html
    TRE: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    print("define_classifier")
    if classifiertype == "SVM": 
        classifier = svm.SVC(kernel="linear") # linear|poly|rbf
    if classifiertype == "KNN": 
        classifier = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    if classifiertype == "TRE": 
        classifier = tree.DecisionTreeClassifier()
    if classifiertype == "SGD": 
        classifier = SGDClassifier(loss="log", penalty="l2", shuffle=True)
    return classifier
    

def get_metadata(metadatafile, target): 
    """
    Get a list of labels for the target category
    """
    print("get_metadata")
    with open(metadatafile, "r") as InFile: 
        rawmetadata = pd.DataFrame.from_csv(InFile, sep=";")
        labels = rawmetadata.loc[:, target]
        idnos = rawmetadata.loc[:,"idno"]
        metadata = pd.concat([idnos, labels], axis=1)
        #print(metadata.head())        
        return metadata
        

def get_countsdata(taggedfolder): 
    print("get_lemmadata") 
    countsdata = pd.DataFrame()
    for file in glob.glob(taggedfolder + "*.csv"): 
        idno, lemmatext = get_lemmatext(file)
        stoplist = [".", ",", ";", "?", "!", ":", "(", ")", " "]
        lemmatext = [lemma for lemma in lemmatext if lemma not in stoplist]
        lemmacounts = pd.Series(dict(Counter(lemmatext)), name="idno")
        #print(lemmacounts.head())
        countsdata[idno] = lemmacounts
    countsdata = countsdata.T
    countsdata.fillna(0, inplace=True)
    countsdata = countsdata.divide(countsdata.sum(axis=0))
    countsdata = countsdata.subtract(countsdata.mean(axis=0)).divide(countsdata.std(axis=0))
    countsdata = countsdata.sort_index(axis=0)
    #print(countsdata.head())
    return countsdata   
    
    
def classify_withwords(countsdata, mfw, metadata, classifiertypes): 
    """
    Classify items using SVM and evaluate accuracy.
    """
    print("classify")
    countsdata = countsdata.iloc[:,0:mfw]
    data = np.array(countsdata)
    print
    labels = metadata.loc[:,"subgenre"].values
    labels = np.array(labels)
    #print(labels[0:5], len(labels), set(labels))
    results = [mfw]
    index = ["mfw"]
    for classifiertype in classifiertypes: 
        classifier = define_classifier(classifiertype)
        accuracy = cross_validation.cross_val_score(classifier, data, labels, cv=10, scoring="accuracy")
        print("Mean accuracy with "+str(classifiertype)+": %0.3f (+/- %0.3f)" % (accuracy.mean(), accuracy.std() * 2))       
        results.append("{:01.3f}".format(accuracy.mean()))
        index.append(str(classifiertype))
        
        # Analyze support vectors (useful features) 
        classifier.fit(data, labels)
        #classifier.predict(data)
        #print("support_vectors", classifier.support_vectors_)
        #print("indices of suppport vectors", classifier.support_)
        #print("classes", classifier.classes_)
        #print("num_sv_per_class", classifier.n_support_)
        #print("class weight", classifier.class_weight)
        coefs = classifier.coef_[0]
        features = countsdata.columns.values
        matrix = np.matrix([features, coefs])
        featureweights = pd.DataFrame(data=matrix).T
        featureweights.columns = ['lemma', 'weight']
        featureweights = featureweights.sort_values(by="weight", ascending=False)
        print(featureweights.iloc[:10,:])
        print(featureweights.iloc[-10:,:])
        
        bestfeatures = featureweights.loc[:,"lemma"]
        bestfeatures = bestfeatures.values[:10] + bestfeatures.values[-10:]
        print(bestfeatures) 
        countsdata.filter(items=bestfeatures)
        print(countsdata.head())
        
        
            
        
        
        
    #resultsname = str(mfw)
    #results = pd.Series(results, index=index, name=resultsname)
    #print(results)
    
    # Analyze support vectors (useful features) 
    
    
    
    return results

    

def classify(taggedfolder, metadatafile, target, classifiertypes, resultsfile): 
    """
    Classify plays (e.g. into subgenres) based on their word frequencies.
    Finds out how well data can be classified with given data. 
    """
    print("Classify by "+target+".") 
    metadata = get_metadata(metadatafile, target)
    countsdata = get_countsdata(taggedfolder)
    for mfw in [1500]:
        results = classify_withwords(countsdata, mfw, metadata, classifiertypes)
    print("Done.")




#########################################
# Plot topic-based classifier results
#########################################

import pygal

my_style = pygal.style.Style(
  background='white',
  plot_background='white',
  font_family = "FreeSans",
  title_font_size = 16,
  legend_font_size = 14,
  label_font_size = 12,
  guide_stroke_dasharray = "0.5,0.5",
  colors=["darkblue","darkcyan","dodgerblue","purple", "#071746"])


def get_topicresults(TopicResultsFile): 
    with open(TopicResultsFile, "r") as InFile: 
        TopicResults = pd.read_csv(InFile, sep=",")
        TopicResults.rename(columns={"Unnamed: 0": "label"}, inplace=True)
        TopicResults.set_index("label", drop=True, inplace=True)
        #print(TopicResults)
        return TopicResults

def get_topicdata(TopicResults, Algorithm):
    #TopicResults.sort_values(Algorithm, ascending=True, inplace=True)
    TopicResults.sort_index(axis=0, ascending=True, inplace=True)
    Data = list(TopicResults.loc[:,Algorithm])
    Labels = TopicResults.index.values
    #print(Labels, Data)
    return Data, Labels


def make_topicplot(TopicResults, GraphFolder): 
    print("make_plot")
    plot = pygal.Line(title = "Classifier performance on topic data" , 
                      x_title = "Input data parameters \n(number of topics, optimize interval)" ,
                      y_title = "Mean accuracy \n(10-fold cross-validation)",
                      legend_at_bottom = True,
                      legend_at_bottom_columns = 5,
                      legend_box_size = 16,
                      style = my_style,
                      x_label_rotation=75,
                      show_x_guides=True,
                      interpolate='cubic')
    for Algorithm in ["SVM", "SGD", "KNN", "TRE"]: 
        Data, Labels = get_topicdata(TopicResults, Algorithm)
        plot.x_labels = Labels
        plot.add(Algorithm, Data, stroke_style={"width": 2,}, dots_size=2)        
    for Algorithm in ["average"]: 
        Data, Labels = get_topicdata(TopicResults, Algorithm)
        plot.add(Algorithm, Data, stroke_style={"width": 2, "dasharray" : "1,1"}, dots_size=2)        
    plot.render_to_file(GraphFolder+"classify-performance_topics.svg")


def plot(WordResultsFile,
         TopicResultsFile,
         GraphFolder): 
    """
    Function to make plots from the classification performance data.
    """
    TopicResults = get_topicresults(TopicResultsFile)
    TopicData = make_topicplot(TopicResults, GraphFolder)

    

































        


