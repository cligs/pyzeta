#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: pyzeta.py
# author: #cf
# version: 0.2.0


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


# import itertools
# import shutil
# from sklearn.decomposition import PCA

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


def run_treetagger(text):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG="fr")
    tagged = tagger.tag_text(text)
    return tagged


def save_tagged(taggedfolder, filename, tagged):
    taggedfilename = taggedfolder + "/" + filename + ".csv"
    with open(taggedfilename, "w") as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for item in tagged:
            item = re.split("\t", item)
            writer.writerow(item)


def prepare(plaintextfolder, taggedfolder):
    print("--prepare")
    if not os.path.exists(taggedfolder):
        os.makedirs(taggedfolder)
    for file in glob.glob(plaintextfolder + "*.txt"):
        filename = get_filename(file)
        text = read_plaintext(file)
        tagged = run_treetagger(text)
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
        onemetadata = metadata[metadata[contrast[0]].isin([contrast[1]])]
        twometadata = metadata[metadata[contrast[0]].isin([contrast[2]])]
        onelist = list(onemetadata.loc[:, "idno"])
        twolist = list(twometadata.loc[:, "idno"])
        # print(onelist, twolist)
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
    global features
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
    Zeta = proportion in group one + (1 - proportion in group two) -1
    """
    print("--calculate_zetas")
    # Calculate the proportions by dividing the row-wise sums by the number of segments
    allfeaturecountsone["docpropone"] = np.divide(np.sum(allfeaturecountsone, axis=1), len(allfeaturecountsone.columns))
    allfeaturecountstwo["docproptwo"] = np.divide(np.sum(allfeaturecountstwo, axis=1), len(allfeaturecountstwo.columns))
    zetascoredata = pd.concat([allfeaturecountsone.loc[:, "docpropone"], allfeaturecountstwo.loc[:, "docproptwo"]],
                              axis=1, join="outer")
    zetascoredata = zetascoredata.fillna(0)
    # The next line contains the actual zeta score calculation
    zetascoredata["zetascores"] = zetascoredata.loc[:, "docpropone"] + (1 - zetascoredata.loc[:, "docproptwo"]) - 1
    zetascoredata = zetascoredata.sort_values("zetascores", ascending=False)
    # print(zetascoredata.head(5))
    # print(zetascoredata.tail(5))
    return zetascoredata


def zeta(taggedfolder, metadatafile, contrast,
         datafolder, resultsfolder, seglength,
         pos, forms, stoplistfile):
    """
    Main coordinating function for "pyzeta.zeta"
    Python implementation of Craig's Zeta. 
    Status: proof-of-concept quality.
    """

    # Generate necessary file and folder names
    contraststring = contrast[1] + "-" + contrast[2]
    parameterstring = str(seglength) + "-" + forms + "-" + str(pos)
    segsfolder = datafolder + contraststring + "_" + parameterstring + "/"
    zetascorefile = resultsfolder + contraststring + "_" + parameterstring + ".csv"
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


# =================================
# Functions: plot zetadata
# =================================

zeta_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family="FreeSans",
    title_font_size=20,
    legend_font_size=16,
    label_font_size=12)


def get_zetadata(zetascorefile, numwords):
    with open(zetascorefile, "r") as infile:
        zetadata = pd.DataFrame.from_csv(infile, sep="\t")
        # print(zetadata.head())
        zetadata.drop(["docpropone", "docproptwo"], axis=1, inplace=True)
        zetadata.sort_values("zetascores", ascending=False, inplace=True)
        zetadatahead = zetadata.head(numwords)
        zetadatatail = zetadata.tail(numwords)
        zetadata = zetadatahead.append(zetadatatail)
        zetadata = zetadata.reset_index(drop=False)
        # print(zetadata.head())
        return zetadata


def plot_zetadata(zetadata, contraststring, zetaplotfile, numwords):
    plot = pygal.HorizontalBar(style=zeta_style,
                               print_values=False,
                               print_labels=True,
                               show_legend=False,
                               range=(-1, 1),
                               title=("Kontrastive Analyse mit Zeta\n (" +
                                      str(contraststring) + ")"),
                               x_title="Zeta-Score",
                               y_title="Je " + str(numwords) + " Worte pro Sammlung"
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
        elif zetadata.iloc[i, 1] > 0:
            color = "#4d804d"
        elif zetadata.iloc[i, 1] < -0.8:
            color = "#0066ff"
        elif zetadata.iloc[i, 1] < -0.7:
            color = "#196be6"
        elif zetadata.iloc[i, 1] < -0.6:
            color = "#3370cc"
        elif zetadata.iloc[i, 1] < -0.5:
            color = "#4d75b3"
        elif zetadata.iloc[i, 1] < 0:
            color = "#60799f"
        else:
            color = "DarkGrey"
        plot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 1], "label": zetadata.iloc[i, 0], "color": color}])
    plot.render_to_file(zetaplotfile)


def plot_scores(numwords, contraststring, parameterstring, resultsfolder):
    print("--plot_scores")
    # Define some filenames
    zetascorefile = resultsfolder + contraststring + "_" + parameterstring + ".csv"
    zetaplotfile = resultsfolder + "zetascores_" + contraststring + "_" + parameterstring + ".svg"
    # Get the data and plot it
    zetadata = get_zetadata(zetascorefile, numwords)
    plot_zetadata(zetadata, contraststring, zetaplotfile, numwords)


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
                    title="Distribution of types",
                    x_title="Proportion of types in " + str(contrast[1]),
                    y_title="Proportion of types in " + str(contrast[2]))
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
    zetascorefile = resultsfolder + contraststring + "_" + parameterstring + ".csv"
    typescatterfile = (resultsfolder + "typescatter_" + contraststring + "_"
                       + parameterstring + "_" + str(numfeatures) + ".svg")
    scores = get_scores(zetascorefile, numfeatures)
    thetypes, propsone, propstwo, zetas = make_data(scores)
    make_typesplot(thetypes, propsone, propstwo, zetas, numfeatures, cutoff, contrast, typescatterfile)
    print("Done.")

# ==============================================
# Threeway comparison
# ==============================================


# def get_threewayscores(zetafile, numfeatures):
#     with open(zetafile, "r") as infile:
#         zetascores = pd.DataFrame.from_csv(infile)
#         scores = zetascores.head(numfeatures)
#         # print(scores.head())
#         return scores
#
#
# def get_features(scores):
#     features = list(scores.loc[:, "type"])
#     # print("features", features)
#     return features
#
#
# def make_three_filelist(metadatafile, threecontrast):
#     """
#     Based on the metadata, create three lists of files, each from one group.
#     The category to check and the two labels are found in threeContrast.
#     """
#     with open(metadatafile, "r") as infile:
#         metadata = pd.DataFrame.from_csv(infile, sep=";")
#         threecontrast = threecontrast[0]
#         onemetadata = metadata[metadata[threecontrast[0]].isin([threecontrast[1]])]
#         twometadata = metadata[metadata[threecontrast[0]].isin([threecontrast[2]])]
#         threemetadata = metadata[metadata[threecontrast[0]].isin([threecontrast[3]])]
#         onelist = list(onemetadata.loc[:, "idno"])
#         twolist = list(twometadata.loc[:, "idno"])
#         threelist = list(threemetadata.loc[:, "idno"])
#         # print(oneList, twoList, threeList)
#         print("---", len(onelist), len(twolist), len(threelist), "texts")
#         return onelist, twolist, threelist
#
#
# def get_freqs(prepared):
#     freqsall = Counter(prepared)
#     # print(freqsall)
#     return freqsall
#
#
# def select_freqs(freqsall, features, textlength, textname):
#     freqssel = dict((key, freqsall[key] / textlength) for key in features)
#     freqssel = pd.Series(freqssel, name=textname)
#     # print(freqssel)
#     return freqssel
#
#
# def apply_pca(freqmatrix):
#     pca = PCA(n_components=5, whiten=True)
#     pca.fit(freqmatrix)
#     variance = pca.explained_variance_ratio_
#     transformed = pca.transform(freqmatrix)
#     # print(transformed)
#     print(variance)
#     return transformed, variance
#
#
# def make_2dscatterplot(transformed, components, textnames,
#                        onelist, twolist, threelist,
#                        variance,
#                        mode, forms, pos):
#     components = [components[0] - 1, components[1] - 1]
#     plot = pygal.XY(style=zeta_style,
#                     stroke=False,
#                     show_legend=False,
#                     title="PCA mit distinktiven Features",
#                     x_title="PC" + str(components[0] + 1) + "(" + "{:03.2f}".format(variance[components[0]]) + ")",
#                     y_title="PC" + str(components[1] + 1) + "(" + "{:03.2f}".format(variance[components[1]]) + ")",
#                     )
#     for i in range(0, 391):  # TODO: Derive from number of texts actually used.
#         point = (transformed[i][components[0]], transformed[i][components[1]])
#         if textnames[i] in onelist:
#             mylabel = "comedie"
#             mycolor = "red"
#         elif textnames[i] in twolist:
#             mylabel = "tragedie"
#             mycolor = "blue"
#         elif textnames[i] in threelist:
#             mylabel = "tragicomedie"
#             mycolor = "green"
#         else:
#             mylabel = "ERROR"
#             mycolor = "grey"
#         plot.add(textnames[i], [{"value": point, "label": mylabel, "color": mycolor}])
#     plot.render_to_file(
#         "threeway-2dscatter_" + mode + "-" + forms + "-" + str(pos[0]) + "_PC" + str(components[0] + 1) + "+" + str(
#             components[1] + 1) + ".svg")
#
#
# # Coordinating function
# def threeway(datafolder, zetafile, numfeatures, components,
#              inputfolder, metadatafile, threecontrast,
#              seglength, mode, pos, forms, stoplist):
#     print("--threeway")
#     featuresall = []
#     for contrast in threecontrast[1:4]:
#         zetafile = (datafolder + contrast[1] + "-" + contrast[2] + "_zeta-scores_segs-of-" +
#                     str(seglength) + "-" + mode + "-" + forms + "-" + str(pos[0]) + ".csv")
#         # print(zetaFile)
#         scores = get_threewayscores(zetafile, numfeatures)
#         features = get_features(scores)
#         featuresall.extend(features)
#     # print(featuresall)
#     onelist, twolist, threelist = make_three_filelist(metadatafile, threecontrast)
#     freqmatrix = pd.DataFrame()
#     textnames = []
#     for textfile in glob.glob(inputfolder + "*.txt"):
#         text, textname = read_plaintext(textfile)
#         textnames.append(textname)
#         prepared = prepare_text(text, mode, pos, forms, stoplist)
#         textlength = len(prepared)
#         freqsall = get_freqs(prepared)
#         freqssel = select_freqs(freqsall, featuresall, textlength, textname)
#         freqmatrix[textname] = freqssel
#     print(freqmatrix.shape)
#     transformed, variance = apply_pca(freqmatrix.T)
#     make_2dscatterplot(transformed, components, textnames,
#                        onelist, twolist, threelist,
#                        variance,
#                        mode, forms, pos)
