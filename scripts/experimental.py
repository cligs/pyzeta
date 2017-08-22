#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: complot.py
# author: #cf
# version: 0.1.0


# =================================
# Import statements
# =================================

import os

import pandas as pd
import pygal
from pygal import style
from scipy.stats import kendalltau

from scripts.correlation import calc_rbo as rbo_score

# =================================
# Pygal style
# =================================

zeta_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family="FreeSans",
    title_font_size=18,
    legend_font_size=14,
    label_font_size=12,
    major_label_font_size=12,
    value_font_size=12,
    major_value_font_size=12,
    tooltip_font_size=12,
    opacity_hover=0.2)


# =================================
# Functions: comparisonplot
# =================================


def get_zetadata(resultsfile, comparison, numfeatures):
    with open(resultsfile, "r") as infile:
        alldata = pd.DataFrame.from_csv(infile, sep="\t")
        zetadata = alldata.loc[:, [comparison[0], comparison[1]]]
        zetadata.sort_values(comparison[0], ascending=False, inplace=True)
        zetadata = zetadata.head(numfeatures)
        zetadata = zetadata.reset_index(drop=False)
        # print(zetadata)
        return zetadata


def add_ranks(zetadata, comparison):
    zetadata.sort_values(comparison[0], ascending=False, inplace=True)
    zetadata[comparison[0] + "-ranks"] = zetadata.loc[:, comparison[0]].rank(axis=0, ascending=True)
    zetadata.sort_values(comparison[1], ascending=False, inplace=True)
    zetadata[comparison[1] + "ranks"] = zetadata.loc[:, comparison[1]].rank(axis=0, ascending=True)
    zetadata.sort_values(comparison[0] + "-ranks", ascending=False, inplace=True)
    return zetadata


def make_barchart(zetadata, comparisonplotfile, parameterstring, contraststring, comparison, numfeatures):
    plot = pygal.Bar(style=zeta_style,
                     print_values=False,
                     print_labels=False,
                     show_legend=False,
                     # list(zetadata.loc[:,"index"]),
                     show_x_labels=True,
                     range=(0, numfeatures),
                     title=("Vergleich von Zetawort-Listen nach Rang"),
                     y_title="Inverser Rang der \n" + str(numfeatures) + " distinktivsten Merkmale",
                     x_title="Vergleich von: " + comparison[0] + " (grau) und " + comparison[1] + " (farbig)")
    for i in range(0, numfeatures):
        plot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 3], "color": "darkslategrey"}])
        if zetadata.iloc[i, 3] < zetadata.iloc[i, 4]:
            color = "darkgreen"
        elif zetadata.iloc[i, 3] > zetadata.iloc[i, 4]:
            color = "darkred"
        else:
            color = "darkslategrey"
        plot.add(zetadata.iloc[i, 0], [{"value": zetadata.iloc[i, 4], "color": color}])
        plot.add(zetadata.iloc[i, 0], [{"value": 0, "label": "", "color": "white"}])
    plot.render_to_file(comparisonplotfile)


def comparisonplot(resultsfolder, plotfolder, comparison, numfeatures, segmentlength, featuretype, contrast):
    print("--barchart (comparison)")
    # Define some strings and filenames
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
    comparisonplotfile = plotfolder + "comparisonbarchart_" + parameterstring + "_" + contraststring + "_" + str(
        numfeatures) + "-" + str(comparison[0]) + "-" + str(comparison[1]) + ".svg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    # Get the data and plot it
    zetadata = get_zetadata(resultsfile, comparison, numfeatures)
    zetadata = add_ranks(zetadata, comparison)
    make_barchart(zetadata, comparisonplotfile, parameterstring, contraststring, comparison, numfeatures)


def get_correlation(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast):
    print("--correlation_measures (comparison)")
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"

    # Get the data and compare it
    zetadata = get_zetadata(resultsfile, comparison, numfeatures)
    zetadata = add_ranks(zetadata, comparison)

    resultsfile = resultsfolder + "correlation_" + parameterstring + "_" + contraststring + ".csv"
    with open(resultsfile, "w") as file:
        rank_columns = zetadata.columns[(len(zetadata.columns) - 1) // 2 + 1:]
        for i, zeta_1 in enumerate(rank_columns):
            for zeta_2 in rank_columns[i + 1:]:
                rbo = rbo_score(list(zetadata[zeta_1].values), list(zetadata[zeta_2].values))
                tau = kendalltau(list(zetadata[zeta_1].values), list(zetadata[zeta_2].values))

                file.write("Correlations between %s and %s:\n" % (zeta_1, zeta_2))
                file.write("RBO: %.5f\n" % rbo)
                file.write("Kendalls Tau: %s\n" % str(tau))
                file.write("\n\n\n")
