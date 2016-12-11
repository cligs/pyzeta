#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyzeta.py
# #cf



# =================================
# Import statements
# =================================

import os
import re
import glob
import pandas as pd
from collections import Counter
import treetaggerwrapper
import itertools
import shutil
import pygal
from sklearn.decomposition import PCA



# =================================
# Functions
# =================================


def make_filelist(DataFolder, MetadataFile, Contrast):
    """
    Based on the metadata, create two lists of files, each from one group.
    The category to check and the two labels are found in Contrast.
    """
    with open(MetadataFile, "r") as InFile:
        Metadata = pd.DataFrame.from_csv(InFile, sep=";")
        OneMetadata = Metadata[Metadata[Contrast[0]].isin([Contrast[1]])]
        TwoMetadata = Metadata[Metadata[Contrast[0]].isin([Contrast[2]])]
        OneList = list(OneMetadata.loc[:, "idno"])
        TwoList = list(TwoMetadata.loc[:, "idno"])
        # print(OneList, TwoList)
        print("---", len(OneList), len(TwoList), "texts")
        return OneList, TwoList


def merge_text(DataFolder, List, File):
    """
    Merge all texts from one group into one large text file.
    Creates less loss when splitting.
    """
    with open(File, 'wb') as OutFile:
        PathList = [DataFolder + Item + ".txt" for Item in List]
        for File in PathList:
            try:
                with open(File, 'rb') as ReadFile:
                    shutil.copyfileobj(ReadFile, OutFile)
            except:
                print("exception.")


def read_file(File):
    """
    Read one text file per partition.
    """
    FileName, Ext = os.path.basename(File).split(".")
    with open(File, "r") as InFile:
        Text = InFile.read()
    # print(Text)
    return Text, FileName


def prepare_text(Text, Mode, Pos, Forms, Stoplist):
    """
    Takes a text in string format and transforms and filters it. 
    Makes it lowercase, splits into tokens, discards tokens of length 1.
    Alternatively, applies POS-tagging and selection of specific POS.
    Returns a list. 
    """
    if Mode == "plain": 
        Prepared = Text.lower()
        Prepared = re.split("\W", Prepared)
        Prepared = [Token for Token in Prepared if len(Token) > 1]    
    if Mode == "tag": 
        Tagger = treetaggerwrapper.TreeTagger(TAGLANG="fr")
        print("---tagging")
        Tagged = Tagger.tag_text(Text)
        print("---done tagging")
        Prepared = []
        for Line in Tagged:
            Line = re.split("\t", Line)
            if len(Line) == 3: 
            #print(len(Line), Line)
                if Forms == "lemmas":
                    Prepared.append(Line[2])
                elif Forms == "words": 
                    Prepared.append(Line[0])
                elif Forms == "pos": 
                    Prepared.append(Line[1])
        Prepared = [Token for Token in Prepared if len(Token) > 1]    
    if Mode == "sel": 
        Tagger = treetaggerwrapper.TreeTagger(TAGLANG="fr")
        print("---tagging")
        Tagged = Tagger.tag_text(Text)
        print("---done tagging")
        Prepared = []
        for Line in Tagged:
            Line = re.split("\t", Line)
            if len(Line) == 3: 
            #print(len(Line), Line)
                if Line[1][0:2] in Pos:
                    if Forms == "lemmas":
                        Prepared.append(Line[2])
                    elif Forms == "words": 
                        Prepared.append(Line[0])
                    elif Forms == "pos": 
                        Prepared.append(Line[1])
    if Mode == "posbigrams": 
        Tagger = treetaggerwrapper.TreeTagger(TAGLANG="fr")
        print("---tagging")
        Tagged = Tagger.tag_text(Text)
        print("---done tagging")
        Prepared = []
        for i in range(0,len(Tagged)-1): 
            Line = re.split("\t", Tagged[i])
            NextLine = re.split("\t", Tagged[i+1])
            Prepared.append(Line[1]+"-"+NextLine[1])
    if Mode == "wordbigrams": 
        Text = Text.lower()
        Text = re.split("\W", Text)
        Text = [Token for Token in Text if len(Token) > 1]    
        Prepared = []
        for i in range(0,len(Text)-1): 
            Prepared.append(Text[i]+"-"+Text[i+1])
    Prepared = [Item.lower() for Item in Prepared if Item not in Stoplist]
    #print(Prepared[0:50])
    return Prepared


def save_seg(Seg, SegFile, SegsFolder):
    """
    Function to save one segment to disk for sequential reading.
    """
    with open(SegsFolder + SegFile, "w") as OutFile:
        OutFile.write(Seg)


def segment_text(Prepared, SegLength, Filename, SegsFolder):
    """
    Splits the whole text document into segments of fixed length; discards rest. 
    Also, reduces each segment to the set of different words in the segment. 
    """
    NumSegs = int(len(Prepared) / SegLength)
    # print("text length (prepared)", len(Prepared))
    # print("number of segments", NumSegs)
    for i in range(0, NumSegs):
        Seg = Prepared[i * SegLength:(i + 1) * SegLength]
        # print(len(Seg))
        Seg = list(set(Seg))
        # print(len(Seg))
        Seg = "\t".join(Seg)
        SegFile = Filename + "{:04d}".format(i) + ".txt"
        save_seg(Seg, SegFile, SegsFolder)
    return NumSegs


def get_types(OnePrepared, TwoPrepared, Threshold):
    """
    Merges all prepared text and extracts the types with their frequency (Counter). 
    Filters the list of types based on their frequency and length in chars.
    A high frequency threshold may speed things up but information is lost. 
    """
    Types = Counter()
    Types.update(OnePrepared)
    Types.update(TwoPrepared)
    # print(Types)
    Types = {k: v for (k, v) in Types.items() if v > Threshold and len(k) > 1}
    # print(Types)
    # Set all values to zero.
    Types = dict.fromkeys(Types, 0)
    # print("number of types in collection (filtered)", len(list(Types.keys())))
    # print(list(itertools.islice(Types.items(), 0, 5)))
    return Types


def check_types(SegsPath, Types, NumSegs):
    """
    For each text segment in one group: 
    1. Read the file and split on the tab
    2. For each Type in the list of all Types, check whether it exists in the file.
    3. If it does, increase the value in the dict for this type by one.
    At the end, divide all dict values by the number of segments. 
    """
    Types = dict.fromkeys(Types, 0)
    for SegFile in glob.glob(SegsPath):  # TODO: this part is really slow ###
        # print("SegFile:", SegFile)
        with open(SegFile, "r") as InFile:
            Seg = InFile.read()
            Seg = re.split("\t", Seg)
            for Type in Types:
                if Type in Seg:
                    Types[Type] = Types[Type] + 1
    Props = {k: v / NumSegs for k, v in Types.items()}
    return Props


def get_zetas(Types, OneProps, TwoProps, ZetaFile):
    """
    Perform the actual Zeta calculation.
    Zeta = Proportion in Group One + (1-Proportion in Group 2) -1
    """
    AllResults = []
    for Type in Types:
        try:
            OneProp = OneProps[Type]
        except:
            OneProp = 0
        try:
            TwoProp = TwoProps[Type]
        except:
            TwoProp = 0
        Zeta = OneProp + (1 - TwoProp) - 1
        Result = {"type": Type, "one-prop": OneProp, "two-prop": TwoProp, "zeta": Zeta}
        AllResults.append(Result)
    AllResults = pd.DataFrame(AllResults)
    AllResults = AllResults[["type", "one-prop", "two-prop", "zeta"]]
    AllResults = AllResults.sort_values("zeta", ascending=False)
    print(AllResults.head(10))
    print(AllResults.tail(10))
    with open(ZetaFile, "w") as OutFile:
        AllResults.to_csv(OutFile)


# =================================
# Main coordinating function
# =================================

def zeta(WorkDir, InputFolder,
         MetadataFile, Contrast,
         DataFolder,
         SegLength, Threshold,
         Mode, Pos, Forms, Stoplist):
    """
    Python implementation of Craig's Zeta. 
    Status: proof-of-concept quality.
    """
    # Generate necessary file and folder names
    OneFile = DataFolder + Contrast[1] + ".txt"
    TwoFile = DataFolder + Contrast[2] + ".txt"
    SegsFolder = DataFolder + Contrast[1] + "-" + Contrast[2] + "_segs-of-" + str(
        SegLength) + "-" + Mode + "-" + Forms + "-" + str(Pos[0]) + "/"
    ZetaFile = DataFolder + Contrast[1] + "-" + Contrast[2] + "_zeta-scores_segs-of-" + str(
        SegLength) + "-" + Mode + "-" + Forms + "-" + str(Pos[0]) + ".csv"
    # Create necessary folders
    if not os.path.exists(DataFolder):
        os.makedirs(DataFolder)
    if not os.path.exists(SegsFolder):
        os.makedirs(SegsFolder)
    # Generate list of files for the two groups
    print("--generate list of files")
    OneList, TwoList = make_filelist(InputFolder, MetadataFile, Contrast)
    # Merge text files into two input files
    print("--merge_text (one and two)")
    merge_text(InputFolder, OneList, OneFile)
    merge_text(InputFolder, TwoList, TwoFile)
    # Load both text files       
    print("--read_file (one and two)")
    OneText, OneFileName = read_file(OneFile)
    TwoText, TwoFileName = read_file(TwoFile)
    # Prepare both text files
    print("--prepare_text (one)")
    OnePrepared = prepare_text(OneText, Mode, Pos, Forms, Stoplist)
    print("--prepare_text (two)")
    TwoPrepared = prepare_text(TwoText, Mode, Pos, Forms, Stoplist)
    # Segment both text files
    print("--segment_text (one and two)")
    NumSegsOne = segment_text(OnePrepared, SegLength, OneFileName, SegsFolder)
    NumSegsTwo = segment_text(TwoPrepared, SegLength, TwoFileName, SegsFolder)
    print("  Number of segments (one, two)", NumSegsOne, NumSegsTwo)
    # Extract the list of selected types 
    print("--get_types (one)")
    Types = get_types(OnePrepared, TwoPrepared, Threshold)
    print("  Number of types", len(list(Types.keys())))
    # Check in how many segs each type is (one)
    print("--check_types (one)")
    OneProps = check_types(SegsFolder + Contrast[1] + "*.txt", Types, NumSegsOne)
    # Extract the list of selected types (repeat)
    print("--get_types (two)")
    Types = get_types(OnePrepared, TwoPrepared, Threshold)
    # Check in how many segs each type is (two)
    print("--check_types (two)")
    TwoProps = check_types(SegsFolder + Contrast[2] + "*.txt", Types, NumSegsTwo)
    # Calculate zeta for each type
    print("--get_zetas")
    get_zetas(Types, OneProps, TwoProps, ZetaFile)


# =================================
# Visualize zeta data
# =================================

zeta_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family="FreeSans",
    title_font_size=20,
    legend_font_size=16,
    label_font_size=12)


def get_zetadata(zetafile, numwords):
    with open(zetafile, "r") as infile:
        zetadata = pd.DataFrame.from_csv(infile)
        # print(ZetaData.head())
        zetadata.drop(["one-prop", "two-prop"], axis=1, inplace=True)
        zetadata.sort_values("zeta", ascending=False, inplace=True)
        zetadatahead = zetadata.head(numwords)
        zetadatatail = zetadata.tail(numwords)
        zetadata = zetadatahead.append(zetadatatail)
        zetadata = zetadata.reset_index(drop=True)
        # print(zetadata)
        return zetadata


def plot_zetadata(zetadata, contrast, plotfile, numwords):
    plot = pygal.HorizontalBar(style=zeta_style,
                               print_values=False,
                               print_labels=True,
                               show_legend=False,
                               range=(-1, 1),
                               title=("Kontrastive Analyse mit Zeta\n (" +
                                      contrast[2] + " vs " + contrast[1] + ")"),
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
    plot.render_to_file(plotfile)


def plot_zeta(zetafile,
              numwords,
              contrast,
              plotfile):
    print("--plot_zeta")
    zetadata = get_zetadata(zetafile, numwords)
    plot_zetadata(zetadata, contrast, plotfile, numwords)




# ==============================================
# Scatterplot of types
# ==============================================


def get_scores(ZetaFile, nFeatures):
    with open(ZetaFile, "r") as InFile:
        ZetaScores = pd.DataFrame.from_csv(InFile)
        posScores = ZetaScores.head(nFeatures)
        negScores = ZetaScores.tail(nFeatures)
        Scores = pd.concat([posScores, negScores])
        #print(Scores.head())
        return Scores


def make_data(Scores):
    Types = list(Scores.loc[:, "type"])
    OnePs = list(Scores.loc[:, "one-prop"])
    TwoPs = list(Scores.loc[:, "two-prop"])
    Zetas = list(Scores.loc[:, "zeta"])
    return Types, OnePs, TwoPs, Zetas


def make_typesplot(Types, OnePs, TwoPs, Zetas, nFeatures, cutoff, Contrast, ScatterFile):
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range = (0,1),
                    title = "Distribution of types",
                    x_title = "Proportion of types in "+str(Contrast[1]),
                    y_title = "Proportion of types in "+str(Contrast[2]))
    for i in range(0, nFeatures*2):
        if Zetas[i] > cutoff:
            color = "green"
            size = 4
        elif Zetas[i] < -cutoff:
            color = "blue"
            size = 4
        else:
            color = "grey"
            size = 3
        plot.add(str(Types[i]), [{"value":(OnePs[i], TwoPs[i]), "label": "zeta "+str(Zetas[i]), "color": color, "node": {"r":size}}])
    plot.render_to_file(ScatterFile)


def plot_types(ZetaFile, nFeatures, cutoff, Contrast, ScatterFile):
    """
    Function to make a scatterplot with the type proprtion data.
    """
    print("--plot_types")
    Scores = get_scores(ZetaFile, nFeatures)
    Types, OnePs, TwoPs, Zetas = make_data(Scores)
    make_typesplot(Types, OnePs, TwoPs, Zetas, nFeatures, cutoff, Contrast, ScatterFile)
    print("Done.")






# ==============================================
# Threeway comparison
# ==============================================


def get_threewayscores(zetafile, numfeatures):
    with open(zetafile, "r") as infile:
        zetascores = pd.DataFrame.from_csv(infile)
        scores = zetascores.head(numfeatures)
        # print(scores.head())
        return scores


def get_features(scores):
    features = list(scores.loc[:,"type"])
    # print("features", features)
    return features


def make_three_filelist(metadatafile, threecontrast):
    """
    Based on the metadata, create three lists of files, each from one group.
    The category to check and the two labels are found in threeContrast.
    """
    with open(metadatafile, "r") as infile:
        metadata = pd.DataFrame.from_csv(infile, sep=";")
        threecontrast = threecontrast[0]
        onemetadata = metadata[metadata[threecontrast[0]].isin([threecontrast[1]])]
        twometadata = metadata[metadata[threecontrast[0]].isin([threecontrast[2]])]
        threemetadata = metadata[metadata[threecontrast[0]].isin([threecontrast[3]])]
        onelist = list(onemetadata.loc[:, "idno"])
        twolist = list(twometadata.loc[:, "idno"])
        threelist = list(threemetadata.loc[:, "idno"])
        # print(oneList, twoList, threeList)
        print("---", len(onelist), len(twolist), len(threelist), "texts")
        return onelist, twolist, threelist


def get_freqs(prepared):
    freqsall = Counter(prepared)
    # print(freqsall)
    return freqsall


def select_freqs(freqsall, features, textlength, textname):
    freqssel = dict((key, freqsall[key]/textlength) for key in features)
    freqssel = pd.Series(freqssel, name=textname)
    # print(freqssel)
    return freqssel


def apply_pca(freqmatrix):
    pca = PCA(n_components=5, whiten=True)
    pca.fit(freqmatrix)
    variance = pca.explained_variance_ratio_
    transformed = pca.transform(freqmatrix)
    # print(transformed)
    print(variance)
    return transformed, variance


def make_2dscatterplot(transformed, components, textnames,
                       onelist, twolist, threelist,
                       variance,
                       mode, forms, pos):
    components = [components[0] -1, components[1] - 1]
    plot = pygal.XY(style=zeta_style,
                    stroke=False,
                    show_legend=False,
                    title = "PCA mit distinktiven Features",
                    x_title = "PC" +str(components[0]+1)+ "("+"{:03.2f}".format(variance[components[0]])+")",
                    y_title = "PC" +str(components[1]+1)+ "("+"{:03.2f}".format(variance[components[1]])+")",
                    )
    for i in range(0, 391):   # TODO: Derive from number of texts actually used.
        point = (transformed[i][components[0]], transformed[i][components[1]])
        if textnames[i] in onelist:
            mylabel = "comedie"
            mycolor = "red"
        elif textnames[i] in twolist:
            mylabel = "tragedie"
            mycolor = "blue"
        elif textnames[i] in threelist:
            mylabel = "tragicomedie"
            mycolor = "green"
        else:
            mylabel = "ERROR"
            mycolor = "grey"
        plot.add(textnames[i], [{"value":point, "label":mylabel, "color":mycolor}])
    plot.render_to_file("threeway-2dscatter_" + mode + "-" + forms + "-" + str(pos[0]) + "_PC" + str(components[0]+1) + "+" + str(components[1]+1) +".svg")


# Coordinating function
def threeway(datafolder, zetafile, numfeatures, components,
             inputfolder, metadatafile, threecontrast,
             seglength, mode, pos, forms, stoplist):
    print("--threeway")
    featuresall = []
    for contrast in threecontrast[1:4]:
        zetafile = (datafolder + contrast[1] + "-" + contrast[2] + "_zeta-scores_segs-of-" +
                    str(seglength) + "-" + mode + "-" + forms + "-" + str(pos[0]) + ".csv")
        #print(zetaFile)
        scores = get_threewayscores(zetafile, numfeatures)
        features = get_features(scores)
        featuresall.extend(features)
    #print(featuresall)
    onelist, twolist, threelist = make_three_filelist(metadatafile, threecontrast)
    freqmatrix = pd.DataFrame()
    textnames = []
    for textfile in glob.glob(inputfolder + "*.txt"):
        text, textname = read_file(textfile)
        textnames.append(textname)
        prepared = prepare_text(text, mode, pos, forms, stoplist)
        textlength = len(prepared)
        freqsall = get_freqs(prepared)
        freqssel = select_freqs(freqsall, featuresall, textlength, textname)
        freqmatrix[textname] = freqssel
    print(freqmatrix.shape)
    transformed, variance = apply_pca(freqmatrix.T)
    make_2dscatterplot(transformed, components, textnames,
                       onelist, twolist, threelist,
                       variance,
                       mode, forms, pos)
