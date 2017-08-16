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





        


