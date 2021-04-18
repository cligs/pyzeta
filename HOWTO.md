# pyzeta: how-to

[![DOI](https://zenodo.org/badge/76167647.svg)](https://zenodo.org/badge/latestdoi/76167647)

## Purpose of this document

This document contains some notes intended to help people use pyzeta.


## What are the requirements?

Requirements:

- Python 3
- Packages pandas, numpy, treetaggerwrapper and pygal
- TreeTagger


## How to install pyzeta?

- Simply download or clone the pyzeta repository
- Make sure you have all the files in the `script`folder, including `run_pyzeta.py`, in a common location on your computer


## How to run pyzeta?

- Open run_pyzeta.py with an IDE such as Geany, Spyder or PyCharm
- Adapt the parameters in run_pyzeta.py to you needs
- Run from the IDE. 
- (Once you have set all parameters in run_pyzeta.py, you can also call it from the Terminal.)


## What is necessary to run the analyses?

The script expects the following as input. See the `sampledata` folder for an example. 

- A folder with plain text files. They need to be in UTF-8 encoding. The files should all be in one folder (here, the `corpus`folder). 
- A metadata file called "metadata.csv" with category information about each file, identified through the column header called "idno" and which contains the filenames (without the extension). The metadata file should be a CSV file, with the "\t" used as the separator character. This metadata file should be at the same level as the `corpus` folder (here, it is in the `sampledata` folder)
- A file with stopwords, called `stoplist.txt`, with one stopword per line. (This can be empty but should be there.)


## What kind of output does pyzeta produce?

The folder `sampledata` contains some examples of what pyzeta produces:

- A folder (`data`) containing the text segments with selected features, as used in the calculation (useful for checking)
- In the folder `results`, a matrix containing the features used with their proportions in each partition and their resulting zeta score
- In the folder `plots`, a plot showing the most distinctive words as a horizontal bar chart and a plot showing the feature distribution as a scatterplot.


## What processes and options are supported?

Currently, the following standard processes are supported:

- Prepare a text collection by tagging it using TreeTagger (pyzeta.prepare; run once per collection)
- For any two partitions of the collection, create a matrix of per-segment type frequencies and calculate the zeta scores for the vocabulary (pyzeta.zeta). There are options to choose word forms or lemmata or POS as features. There is the possibility to filter features based on their POS.
- Visualize the most distinctive words as a horizontal bar chart. (pyzeta.plot_scores)
- Visualize the feature distribution as a scatterplot (pyzeta.plot_types)

The following experimental functions are present (in the previous release, and not really supported):

- Visualize the relation between three partitions based on type proportions in two partitions (pyzeta.threeway)
- PCA for three partitions using distinctive features.


## What parameters are there to control pyzeta behavior?

You can set the following parameters:

- How many word forms should each segment have (500-5000 may be reasonable)
- Which POS should be selected? "all" selects all, "Xy" will select words corresponding to Xy POS tag.
- Which partitions of the data should be contrasted? Indicate the category and the two contrasting labels.
- If you use ["random", "one", "two"] as the value for the `contrast` parameter, the partitions will be built randomly, splitting the collection in equal-sized parts. This is interesting if you want to see how strong your zeta scores really are relative to a random partitioning. (Expanding on this principle, with repeated random measures, is a basis for creating some sort of confidence interval for zeta scores.)


## Where can more background readings about Zeta be found?

For more information on Zeta, see:

- Burrows, John. „All the Way Through: Testing for Authorship in Different Frequency Strata“. Literary and Linguistic Computing 22, Nr. 1 (2007): 27–47. doi:10.1093/llc/fqi067.
- Hoover, David L. „Teasing out Authorship and Style with T-Tests and Zeta“. In Digital Humanities Conference. London, 2010. http://dh2010.cch.kcl.ac.uk/academic-programme/abstracts/papers/html/ab-658.html.
- Schöch, Christof. „Genre Analysis“. In Digital Humanities for Literary Studies: Theories, Methods, and Practices, ed. by James O’Sullivan. University Park: Pennsylvania State Univ. Press, 2017 (to appear).
- Christof Schöch: "Zeta für die kontrastive Analyse literarischer Texte. Theorie, Implementierung, Fallstudie", in: _Quantitative Verfahren in der Literaturwissenschaft. Von einer Scientia Quantitatis zu den Digital Humanities_, ed. Andrea Albrecht, Sandra Richter, Marcel Lepper, Marcus Willand und Toni Bernhart, Berlin: de Gruyter (to appear).

## When using pyzeta for research, how can it be references?

You can either cite the software itself, using the citation suggestion below, or cite the current reference article (ahem, unpublished and in German).

- Christof Schöch. *pyzeta. Python implementation of the Zeta score for contrastive text analysis*. Release 0.3.0. Würzburg: CLIGS, 2017. https://github.com/cligs/pyzeta
- Christof Schöch: "Zeta für die kontrastive Analyse literarischer Texte. Theorie, Implementierung, Fallstudie", in: _Quantitative Verfahren in der Literaturwissenschaft. Von einer Scientia Quantitatis zu den Digital Humanities_, ed. Andrea Albrecht, Sandra Richter, Marcel Lepper, Marcus Willand und Toni Bernhart, Berlin: de Gruyter (to appear).
