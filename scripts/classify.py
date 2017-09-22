from os.path import join
from pathlib import Path

import pandas as pd
import numpy as np
import argparse

from stop_words import get_stop_words

parser = argparse.ArgumentParser()

parser.add_argument("--corpus", default="novelas", type=str, )
parser.add_argument("--num_features", default=20, type=int)

args = parser.parse_args()

# =================================
# Parameters: files and folders
# =================================

# You need to adapt these
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import LinearSVC

corpus = args.corpus
print(corpus)
workdir = "/home/ls6/zehe/remote_python/pyzeta"
dtmfolder = join("/home/ls6/zehe/remote_python/pyzeta/pyzeta-dtms/", corpus, "")

# It is recommended to name your files and folders accordingly
datadir = join(workdir, "data", corpus, "")
metadatafile = join(datadir, "metadata.csv")

# It is recommended not to change these
outputdir = join(workdir, "output", corpus)
taggedfolder = join(outputdir, "tagged", "")
resultsfolder = join(outputdir, "results", "")
classificationfolder = join(outputdir, "classification", "")

segmentlength = 2000
max_num_segments = -1
featuretype = ["lemmata", "all"]

measures = ["sd0", "sd2", "sdX", "sr0", "sr2", "srX", "dd0", "dd2", "ddX", "dr0", "dr2", "drX"]
num_features = args.num_features

assert corpus in ("doyle", "theatre", "novelas"), "Invalid corpus specified: %s" % corpus

if corpus == "doyle":
    language = "english"
    contrast = ["subgenre", "detective", "historical"]  # example for doyle
elif corpus == "novelas":
    contrast = ["continent", "Europe", "America"]  # example for novelas
    language = "spanish"
else:
    contrast = ["subgenre", "tragedie", "comedie"]  # example for theatre
    language = "french"


def get_zetadata(resultsfile, measure, numfeatures, droplist=tuple()):
    with open(resultsfile, "r", encoding="utf-8") as infile:
        alldata = pd.DataFrame.from_csv(infile, sep="\t")
        zetadata = alldata.loc[:, [measure, "docprops1"]]
        zetadata.sort_values(measure, ascending=False, inplace=True)
        zetadata.drop("docprops1", axis=1, inplace=True)
        for item in droplist:
            zetadata.drop(item, axis=0, inplace=True)
        zetadata = zetadata.head(numfeatures).append(zetadata.tail(numfeatures))
        zetadata = zetadata.reset_index(drop=False)
        return zetadata


parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
contraststring = str(contrast[0]) + "_" + str(contrast[2]) + "-" + str(contrast[1])
resultsfile = resultsfolder + "results_" + parameterstring + "_" + contraststring + ".csv"
classificationfile = classificationfolder + "results_" + parameterstring + "_" + contraststring + "_" + str(
    num_features) + ".txt"

print(classificationfile)

with open(classificationfile, "w", encoding="utf-8") as f:
    kfold = KFold(shuffle=False)

    for measure in measures:
        f.write("Using %i most distintice words according to measure %s to classify %s\n" % (
            2 * num_features, measure, corpus))

        data = get_zetadata(resultsfile, measure, num_features)

        zeta_vectorizer = TfidfVectorizer(use_idf=True, token_pattern="[^ ]+")

        all_words_string = " ".join(data["index"])

        zeta_vectorizer.fit([all_words_string])
        assert set(zeta_vectorizer.vocabulary_.keys()) == set(
            data["index"]), "Something went wrong building the vectorizer"  # this should be a unit test

        novel_texts = []
        novel_labels = []
        metadata = pd.read_csv(metadatafile, sep="\t", index_col=0)

        for file in Path(taggedfolder).iterdir():
            novel = pd.read_csv(str(file), encoding="utf-8", sep="\t", index_col=None, names=["word", "tag", "lemma"],
                                quoting=3)
            lemmas = novel["lemma"]
            lemmas[lemmas.isnull()] = novel["word"]
            novel_lemmas = " ".join(lemmas)

            novel_texts.append(novel_lemmas)
            novel_labels.append(metadata[contrast[0]][file.stem])

        novels_tfidf = zeta_vectorizer.transform(novel_texts)

        f.write(str(sorted(zeta_vectorizer.vocabulary_)))
        f.write("\n")

        svm = LinearSVC()
        score = cross_val_score(svm, novels_tfidf, novel_labels, cv=kfold)
        f.write(str(score) + "; Mean: " + str(np.mean(score)))

        f.write("\n")
        f.write(("-" * 20 + "\n") * 3)
        f.write("\n")
        f.flush()

    # baseline: Most frequent words
    f.write("Using %i most frequent words to classify %s\n" % (2 * num_features, corpus))

    novel_texts = []
    novel_labels = []
    metadata = pd.read_csv(metadatafile, sep="\t", index_col=0)

    for file in Path(taggedfolder).iterdir():
        novel = pd.read_csv(str(file), encoding="utf-8", sep="\t", index_col=None, names=["word", "tag", "lemma"],
                            quoting=3)
        lemmas = novel["lemma"]
        lemmas[lemmas.isnull()] = novel["word"]
        novel_lemmas = " ".join(lemmas)

        novel_texts.append(novel_lemmas)
        novel_labels.append(metadata[contrast[0]][file.stem])

    vectorizer = TfidfVectorizer(use_idf=True, max_features=2 * num_features,
                                 stop_words=get_stop_words(language))
    novels_tfidf = vectorizer.fit_transform(novel_texts)
    f.write(str(sorted(vectorizer.vocabulary_)))
    f.write("\n")

    svm = LinearSVC()
    score = cross_val_score(svm, novels_tfidf, novel_labels, cv=kfold)
    f.write(str(score) + "; Mean: " + str(np.mean(score)))

    f.write("\n")
    f.write(("-" * 20 + "\n") * 3)
    f.write("\n")
    f.flush()
