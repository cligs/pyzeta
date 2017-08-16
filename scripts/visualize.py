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
import itertools
import shutil
from sklearn.decomposition import PCA
import random
import math
from sklearn import preprocessing as prp
from sklearn import feature_extraction as fe


# =================================
# Pygal style
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


# =================================
# Functions: plot barchart
# =================================


def get_zetadata(resultsfile, measure, numfeatures):
    with open(resultsfile, "r") as infile:
        alldata = pd.DataFrame.from_csv(infile, sep="\t")
        zetadata = alldata.loc[:, [measure, "docprops1"]]
        zetadata.sort_values(measure, ascending=False, inplace=True)
        zetadata.drop("docprops1", axis=1, inplace=True)
        zetadata = zetadata.head(numfeatures).append(zetadata.tail(numfeatures))
        zetadata = zetadata.reset_index(drop=False)
        return zetadata


def make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures):
    plot = pygal.HorizontalBar(style = zeta_style,
                               print_values = False,
                               print_labels = True,
                               show_legend = False,
                               range = (-1, 1),
                               title = ("Kontrastive Analyse\n("+contraststring+")"),
                               y_title = str(numfeatures) + " distinktive Merkmale",
                               x_title = "Parameter: "+ measure +"-"+ parameterstring)
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


def zetabarchart(segmentlength, featuretype, contrast, measure, numfeatures, resultsfolder, plotfolder):
    print("--barchart (zetascores)")
    # Define some strings and filenames
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "zetaresults.csv"
    zetaplotfile = plotfolder + "zetabarchart_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +".svg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    # Get the data and plot it
    zetadata = get_zetadata(resultsfile, measure, numfeatures)
    make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures)






# ==============================================
# Scatterplot of types
# ==============================================


def get_scores(resultsfile, numfeatures):
    with open(resultsfile, "r") as infile:
        zetascores = pd.DataFrame.from_csv(infile, sep="\t")
        positivescores = zetascores.head(numfeatures)
        negativescores = zetascores.tail(numfeatures)
        scores = pd.concat([positivescores, negativescores])
        return scores


def make_data(scores):
    thetypes = list(scores.index)
    propsone = list(scores.loc[:, "docprops1"])
    propstwo = list(scores.loc[:, "docprops2"])
    zetas = list(scores.loc[:, "origzeta"])
    return thetypes, propsone, propstwo, zetas


def make_typesplot(types, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile):
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range=(0, 1),
                    show_y_guides=True,
                    show_x_guides=True,
                    title="Document proportions and " + "measure",
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


def typescatterplot(numfeatures, cutoff, contrast, segmentlength, featuretype, measure, resultsfolder, plotfolder):
    """
    Function to make a scatterplot with the type proprtion data.
    """
    print("--typescatterplot (types)")
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "zetaresults.csv"
    typescatterfile = plotfolder + "typescatterplot_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +"-"+str(cutoff)+".svg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    scores = get_scores(resultsfile, numfeatures)
    thetypes, propsone, propstwo, zetas = make_data(scores)
    make_typesplot(thetypes, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile)

