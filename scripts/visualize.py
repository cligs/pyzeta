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


def get_zetadata(resultsfile, measure, numfeatures, droplist):
    with open(resultsfile, "r") as infile:
        alldata = pd.DataFrame.from_csv(infile, sep="\t")
        zetadata = alldata.loc[:, [measure, "docprops1"]]
        zetadata.sort_values(measure, ascending=False, inplace=True)
        zetadata.drop("docprops1", axis=1, inplace=True)
        for item in droplist:
            zetadata.drop(item, axis=0, inplace=True)
        zetadata = zetadata.head(numfeatures).append(zetadata.tail(numfeatures))
        zetadata = zetadata.reset_index(drop=False)
        return zetadata


def make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures):
    plot = pygal.HorizontalBar(style = zeta_style,
                               print_values = False,
                               print_labels = True,
                               show_legend = False,
                               range = (-0.5, 0.5),
                               title = ("Contrastive Analysis with Zeta\n("+contraststring+")"),
                               y_title = str(numfeatures) + " distinctive features",
                               x_title = "Parameters: "+ measure +"-"+ parameterstring)
    for i in range(len(zetadata)):
        if zetadata.iloc[i, 1] > 0.8:
            color = "#00cc00"
        if zetadata.iloc[i, 1] > 0.7:
            color = "#14b814"
        if zetadata.iloc[i, 1] > 0.6:
            color = "#29a329"
        elif zetadata.iloc[i, 1] > 0.3:
            color = "#3d8f3d"
        elif zetadata.iloc[i, 1] > 0.2:
            color = "#4d804d"
        elif zetadata.iloc[i, 1] < -0.8:
            color = "#0066ff"
        elif zetadata.iloc[i, 1] < -0.7:
            color = "#196be6"
        elif zetadata.iloc[i, 1] < -0.6:
            color = "#3370cc"
        elif zetadata.iloc[i, 1] < -0.3:
            color = "#4d75b3"
        elif zetadata.iloc[i, 1] < -0.2:
            color = "#60799f"
        else:
            color = "#585858"
        plot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 1], "label": zetadata.iloc[i, 0], "color": color}])
    plot.render_to_file(zetaplotfile)


def zetabarchart(segmentlength, featuretype, contrast, measures, numfeatures, droplist, resultsfolder, plotfolder):
    print("--barchart (zetascores)")
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    for measure in measures:
        # Define some strings and filenames
        zetaplotfile = plotfolder + "zetabarchart_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +"-"+str(measure) + ".svg"
        if not os.path.exists(plotfolder):
            os.makedirs(plotfolder)
        # Get the data and plot it
        zetadata = get_zetadata(resultsfile, measure, numfeatures, droplist)
        make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures)






# ==============================================
# Scatterplot of types
# ==============================================


def get_scores(resultsfile, numfeatures, measure):
    with open(resultsfile, "r") as infile:
        zetascores = pd.DataFrame.from_csv(infile, sep="\t")
        zetascores.sort_values(by=measure, ascending=False, inplace=True)
        positivescores = zetascores.head(numfeatures)
        negativescores = zetascores.tail(numfeatures)
        scores = pd.concat([positivescores, negativescores])
        #print(scores.head())
        return scores


def make_data(scores, measure):
    thetypes = list(scores.index)
    propsone = list(scores.loc[:, "docprops1"])
    propstwo = list(scores.loc[:, "docprops2"])
    zetas = list(scores.loc[:, measure])
    return thetypes, propsone, propstwo, zetas


def make_typesplot(types, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile):
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range=(0, 1),
                    show_y_guides=True,
                    show_x_guides=True,
                    title="Document proportions and " + str(measure),
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
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    typescatterfile = plotfolder + "typescatterplot_" + parameterstring +"_"+ contraststring +"_" +str(numfeatures) +"-" + str(cutoff) +"-"+str(measure)+".svg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    scores = get_scores(resultsfile, numfeatures, measure)
    thetypes, propsone, propstwo, zetas = make_data(scores, measure)
    make_typesplot(thetypes, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile)

