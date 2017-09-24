#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: experimental.py
# author: #cf
# version: 0.1.0


# =================================
# Import statements
# =================================

import os
import pandas as pd
import pygal
from pygal import style
from scipy.stats import kendalltau, spearmanr
from os.path import join

from correlation import calc_rbo as rbo_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

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


def get_zetadata_multiple(resultsfile, comparison, numfeatures):
    with open(resultsfile, "r") as infile:
        alldata = pd.DataFrame.from_csv(infile, sep="\t")
        zetadata = alldata.loc[:, comparison]
        zetadata.sort_values(comparison[0], ascending=False, inplace=True)
        zetadata = zetadata.head(numfeatures)
        zetadata = zetadata.reset_index(drop=False)
        # print(zetadata)
        return zetadata


def add_ranks_multiple(zetadata, comparison):
    zetadata.sort_values(comparison[0], ascending=False, inplace=True)
    for i in range(len(comparison)):
        zetadata[comparison[i] + "-ranks"] = zetadata.loc[:, comparison[i]].rank(axis=0, ascending=True)
        zetadata.sort_values(comparison[i], ascending=False, inplace=True)
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


# =================================
# Functions: RBO correlation
# =================================


def get_correlation(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast):
    print("--correlation_measures (comparison)")
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"

    # Get the data and compare it
    zetadata = get_zetadata_multiple(resultsfile, comparison, numfeatures)
    zetadata = add_ranks_multiple(zetadata, comparison)

    columns = ("Measure 1", "Measure 2", "RBO", "Kendall's Tau", "p-value", "Spearman Rho", "p-value")
    df = pd.DataFrame(columns=columns)

    resultsfile = resultsfolder + "correlation_" + parameterstring + "_" + contraststring + ".csv"
    with open(resultsfile, "w") as file:
        rank_columns = zetadata.columns[(len(zetadata.columns) - 1) // 2 + 1:]
        for i, zeta_1 in enumerate(rank_columns):
            for zeta_2 in rank_columns[i + 1:]:
                rbo = rbo_score(list(zetadata[zeta_1].values), list(zetadata[zeta_2].values))
                tau = kendalltau(list(zetadata[zeta_1].values), list(zetadata[zeta_2].values))
                rho = spearmanr(list(zetadata[zeta_1].values), list(zetadata[zeta_2].values))

                # file.write("Correlations between %s and %s:\n" % (zeta_1, zeta_2))
                # file.write("RBO: %.5f\n" % rbo)
                # file.write("Kendalls Tau: %s\n" % str(tau))
                # file.write("\n\n\n")

                df = df.append(pd.Series(index=columns, data=(zeta_1, zeta_2, rbo, tau[0], tau[1], rho[0], rho[1])),
                               ignore_index=True)

        file.write(df.to_csv(index=False, sep="\t"))



# =================================
# Functions: Compare Zeta variants
# =================================



def make_pca(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder):
    """
    This function creates a PCA from the file with the results.
    """
    # Prepare parameters
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
    # Get the data and compare it
    zetadata = get_zetadata_multiple(resultsfile, comparison, numfeatures)
    zetadata = add_ranks_multiple(zetadata, comparison)
    zetadata = zetadata[comparison].T

    # Calculate PCA
    pca = PCA(n_components=2)
    pca.fit(zetadata)
    pca_results = pca.transform(zetadata)
    x = pca_results[:, 0]
    y = pca_results[:, 1]

    # Make and print plot
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(comparison):
        ax.annotate(txt, (x[i],y[i]))
    ax.grid(True)
    ax.set(xlabel='PC 1', ylabel='PC 2',
       title='PCA der Zeta-Varianten (' + str(numfeatures) + " words)");
    zetaplotfile = plotfolder + "PCA_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +".svg"
    fig.savefig(zetaplotfile)


def make_dendrogram(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder):
    """
    This function creates a dendrogram from the results of the Zeta versions
    """
    # Prepare parameters
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
    # Get the data and compare it
    zetadata = get_zetadata_multiple(resultsfile, comparison, numfeatures)
    zetadata = add_ranks_multiple(zetadata, comparison)
    zetadata = zetadata[comparison].T

    linkage_array = ward(zetadata)
    # Now we plot the dendrogram for the linkage_array containing the distances
    # between clusters

    plt.figure(figsize=(8, len(comparison)-2))

    dendrogram(linkage_array, labels=comparison, orientation="left")

    # Mark the cuts in the tree that signify two or three clusters
    plt.xlabel("Cluster distance")
    plt.ylabel("Zeta variants (short labels)")
    plt.title("Hierarchical Cluster (Ward, " + str(numfeatures) + " words)")

    zetaplotfile = plotfolder + "Dendrogram_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +".svg"
    plt.savefig(zetaplotfile)


def make_tsne(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder):
    """
    This function creates a t-SNE from the file with the results.
    """

    # Prepare parameters
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
    # Get the data and compare it
    zetadata = get_zetadata_multiple(resultsfile, comparison, numfeatures)
    zetadata = add_ranks_multiple(zetadata, comparison)
    zetadata = zetadata[comparison].T

    tsne = TSNE(random_state = 0)
    zeta_tsne = tsne.fit_transform(zetadata)

    x = zeta_tsne[:, 0]
    y = zeta_tsne[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    plt.xlim(zeta_tsne[:, 0].min(), zeta_tsne[:, 0].max())
    plt.ylim(zeta_tsne[:, 1].min(), zeta_tsne[:, 1].max())

    for i, txt in enumerate(comparison):
        ax.annotate(txt, (x[i],y[i]))
    ax.grid(True)
    plt.grid(True)
    plt.xlabel("t-SNE feature 0")
    plt.xlabel("t-SNE feature 1")

    plt.title("t-SNE (" + str(numfeatures) + " words)")

    zetaplotfile = plotfolder + "tSNE_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +".svg"
    plt.savefig(zetaplotfile)

def clustering_kmeans(resultsfolder, comparison, numfeatures, segmentlength, featuretype, contrast, plotfolder,n=4):
    """
    This function creates the clusters from the file with the results.
    """
    # Prepare parameters
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
    # Get the data and compare it
    zetadata = get_zetadata_multiple(resultsfile, comparison, numfeatures)
    zetadata = add_ranks_multiple(zetadata, comparison)
    zetadata = zetadata[comparison].T

    print(zetadata)
    print(zetadata.shape)

    # Create the clusters
    kmeans = KMeans(n_clusters=n)
    # TODO: this step works in Jupyter, but it makes the kernel to die in Spyder...
    kmeans.fit(zetadata)

    kmeans_results = zip(zetadata.index.tolist(),kmeans.labels_)
    resultsfile = resultsfolder + "kmeans="+n+"_" + parameterstring + "_" + contraststring + ".txt"
    print(list(kmeans_results))
    with open(resultsfile, "w") as file:
        file.write(list(kmeans_results))

def cluster_correlation(resultsfolder, segmentlength, featuretype, contrast, plotfolder, comparison):
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
    file = resultsfolder + "correlation_" + parameterstring + "_" + contraststring + ".csv"
    correlation = pd.read_csv(file, encoding="utf-8", sep="\t")
    
    tests = {
    "RBO": 1,
    "Kendall's Tau": 1,
    "p-value": 0,
    "Spearman Rho": 1,
    "p-value.1": 0,
    }

    matrixfolder = join(resultsfolder, "matrix")
    if not os.path.exists(matrixfolder):
        os.makedirs(matrixfolder)
    for test, perfect_corr in tests.items():
        print(test)
        df_symmetrical = pd.DataFrame(columns=comparison, index = comparison)
        #print(df_symmetrical)
        for measure1 in comparison:
            for measure2 in comparison:
                if measure1 == measure2:
                    df_symmetrical.loc[measure1, measure2] = perfect_corr
                else:
                    try: 
                        value = correlation.loc[(correlation['Measure 1'] == measure1+"-ranks") & (correlation['Measure 2'] == measure2+"-ranks")][test].tolist()[0]
                    except:
                        value = correlation.loc[(correlation['Measure 1'] == measure2+"-ranks") & (correlation['Measure 2'] == measure1+"-ranks")][test].tolist()[0]
                    df_symmetrical.loc[measure1, measure2] = value
        file = resultsfolder +"/matrix/"+ test+"_matrix_" + parameterstring + "_" + contraststring + ".csv"
        df_symmetrical.to_csv(file, sep='\t', encoding='utf-8', index=True)
        
        linkage_array = ward(df_symmetrical)
        
        plt.figure(figsize=(8, 6))
    
        dendrogram(linkage_array, labels=df_symmetrical.columns, orientation="left")
    
        # Mark the cuts in the tree that signify two or three clusters
        plt.xlabel("Cluster distance")
        plt.ylabel("Zeta variants")
        plt.title("Clustering based on "+test+" correlations")# (" + str(segmentlength) + " words)")
    
        zetaplotfile = plotfolder + "Dendrogram_correlation_"+test+"_"+parameterstring +"_"+ contraststring +"_" + str(segmentlength) +".svg"
        plt.savefig(zetaplotfile)
