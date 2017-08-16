# pyzeta: how-to

[![DOI](https://zenodo.org/badge/76167647.svg)](https://zenodo.org/badge/latestdoi/76167647)

## Purpose of this document

This document contains some notes intended to help people use pyzeta.


## What are the requirements?

Requirements:

- Python 3
- Packages pandas, numpy, treetaggerwrapper and pygal
- Also, TreeTagger must be installed


## How to install pyzeta?

- Simply download or clone the pyzeta repository


## How to run pyzeta?

- Open run_pyzeta.py with an IDE such as Geany, Spyder or PyCharm
- Adapt the parameters in run_pyzeta.py to you needs
- Run from the IDE. 


## What is necessary to run the analyses?

The script expects the following as input. See the `sampledata` folder for an example. 

- A folder (here called `corpus`) with plain text files. They need to be in UTF-8 encoding. The files should all be in one folder. 
- A metadata file with category information about each file, identified through "idno" = filename. The metadata file should be a CSV file, with the ";" used as the separator character.
- A file with stopwords, one per line


## What kind of output does pyzeta produce?

The folder `sampledata` also contains some examples of what pyzeta produces:

- The `data` folder contains the tagged text files as well as the text segments with selected features and the document-term-matrices
- The `results`folder contains a matrix with a number of different scores for each word
- The `plots` folder has several types of plots, e.g. the most distinctive words as a horizontal bar chart or the feature distribution as a scatterplot.


## What processes and options are supported?

Currently, the following standard processes are supported:

- Preprocess a text collection by tagging it using TreeTagger (preprocess.py; run once per collection)
- Depending on some settings, create a document-term matrix of per-segment type frequencies (prepare.py). There are options to choose word forms or lemmata or POS as features. There is the possibility to filter features based on their POS.
- Calculate the zeta scores for the vocabulary (calculate.py). Several types of zeta variants are calculated.
- Visualize the results (visualize.py - not moved to next yet)


## What parameters are there to control pyzeta behavior?

You can set the following parameters:

- How many word forms should each segment have (500-5000 may be reasonable)
- Which POS should be selected? "all" selects all, "Xy" will select words corresponding to Xy POS tag.
- Which partitions of the data should be contrasted? Indicate the category and the two contrasting labels.


## Where can more background readings about Zeta be found?

For more information on Zeta, see:

- Burrows, John. „All the Way Through: Testing for Authorship in Different Frequency Strata“. Literary and Linguistic Computing 22, Nr. 1 (2007): 27–47. doi:10.1093/llc/fqi067.
- Hoover, David L. „Teasing out Authorship and Style with T-Tests and Zeta“. In Digital Humanities Conference. London, 2010. http://dh2010.cch.kcl.ac.uk/academic-programme/abstracts/papers/html/ab-658.html.
- Schöch, Christof. „Genre Analysis“. In Digital Humanities for Literary Studies: Theories, Methods, and Practices, ed. by James O’Sullivan. University Park: Pennsylvania State Univ. Press, 2017 (to appear).


## When using pyzeta for research, how can it be references?

You can either cite the software itself, using the citation suggestion below, or cite the current reference article (ahem, unpublished and in German).

- Christof Schöch. *pyzeta. Python implementation of the Zeta score for contrastive text analysis*. Release 0.3.0. Würzburg: CLIGS, 2017. https://github.com/cligs/pyzeta
- Christof Schöch: "Zeta für die kontrastive Analyse literarischer Texte. Theorie, Implementierung, Fallstudie", in: _Quantitative Verfahren in der Literaturwissenschaft. Von einer Scientia Quantitatis zu den Digital Humanities_, ed. Andrea Albrecht, Sandra Richter, Marcel Lepper, Marcus Willand und Toni Bernhart, Berlin: de Gruyter (to appear).
