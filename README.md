# pyzeta

[![DOI](https://zenodo.org/badge/76167647.svg)](https://zenodo.org/badge/latestdoi/76167647)

**The pyzeta scripts are a Python implementation of Craig's Zeta score for contrastive text analysis.**

Currently, the following processes are supported:

- Prepare a text collection by tagging it using TreeTagger (pyzeta.prepare; run once per collection)
- For any two partitions of the collection, create a matrix of per-segment type frequencies and calculate the zeta scores for the vocabulary (pyzeta.zeta). There are options to choose word forms or lemmata or POS as features. There is the possibility to filter features based on their POS.
- Visualize the most distinctive words as a horizontal bar chart. (pyzeta.plot_scores)
- Visualize the feature distribution as a scatterplot (pyzeta.plot_types)
- Currently non-functional: PCA for three partitions using distinctive features.

The script expects the following as input:

- A folder with plain text files
- A metadata file with category information about each file, identified through "idno" = filename
- A file with stopwords, one per line

You can set the following parameters:

- How many word forms should each segment have (500-5000 may be reasonable)
- Which POS should be selected? "all" selects all, "Xy" will select words corresponding to Xy POS tag.
- Which partitions of the data should be contrasted? Indicate the category and the two contrasting labels.

For more information on Zeta, see:

- Burrows, John. „All the Way Through: Testing for Authorship in Different Frequency Strata“. Literary and Linguistic Computing 22, Nr. 1 (2007): 27–47. doi:10.1093/llc/fqi067.
- Hoover, David L. „Teasing out Authorship and Style with T-Tests and Zeta“. In Digital Humanities Conference. London, 2010. http://dh2010.cch.kcl.ac.uk/academic-programme/abstracts/papers/html/ab-658.html.
- Schöch, Christof. „Genre Analysis“. In Digital Humanities for Literary Studies: Theories, Methods, and Practices, ed. by James O’Sullivan. University Park: Pennsylvania State Univ. Press, 2017 (to appear).

Requirements, installation and usage

- Requirements: Python 3 with pandas, numpy, treetaggerwrapper and pygal
- Installation: Simply copy pyzeta.py and run_pyzeta.py to a common location on your computer
- Usage: Open pyzeta.py and run_pyzeta.py with an IDE such as Spyder or PyCharm, adapt the parameters in run_pyzeta.py and run from the IDE. 
