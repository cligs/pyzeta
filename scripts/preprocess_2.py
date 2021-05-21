"""
The "preprocess" module is the first step in the pyzeta pipeline.
This module deals with linguistic annotation of the texts.
Subsequent modules are: prepare, calculate and visualize.
"""

# =================================
# Import statements
# =================================

import spacy
# download spacy via command line python -m spacy download fr_core_news_sm
import os
import re
import csv
import glob
from collections import Counter
from os.path import join
#nlp = spacy.load("fr_core_news_sm")



workdir = "D:/Downloads/roman20/Testset/"
plaintextfolder = join(workdir, "corpus", "")
outputdir = join(workdir, "output")
taggedfolder = join(outputdir, "tagged_2", "")
language = "fr"


def read_plaintext(file):
    """
    reads plaintext files
    """
    with open(file, "r", encoding="utf-8") as infile:
      text = infile.read()
      text = re.sub("’", "'", text)
      return text



def save_tagged(taggedfolder, filename, tagged):
    """
    Takes the spacy output and writes it to a CSV file.
    """
    taggedfilename = taggedfolder + "/" + filename + ".csv"
    with open(taggedfilename, "w", encoding="utf-8") as outfile:
      writer = csv.writer(outfile, delimiter=',')
      for token in tagged:
        token = token.text, token.pos_, token.lemma_
        #print(token)
        writer.writerow(token)
    return writer


def main(plaintextfolder, taggedfolder, language):
    """
    coordinationsfuction
    :param plaintextfolder:
    :param taggedfolder:
    :param language
    """
    if language == "en":
      nlp = spacy.load("en_core_web_sm")
    # french models
    elif language == "fr":
      nlp = spacy.load("fr_core_news_sm")
      nlp.max_length = 10000000
    print("\n--preprocess")
    if not os.path.exists(taggedfolder):
      os.makedirs(taggedfolder)
    counter = 0
    for file in glob.glob(plaintextfolder + "*.txt"):
      filename, ext = os.path.basename(file).split(".")
      counter += 1
      print("next: file", counter, ":", filename)
      text = read_plaintext(file)
      tagged = nlp(text)
      saving = save_tagged(taggedfolder, filename, tagged)
    return saving


main(plaintextfolder, taggedfolder, language)