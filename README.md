# pyzeta

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.597354.svg)](https://doi.org/10.5281/zenodo.597354)



**The pyzeta scripts are a Python implementation of Craig's Zeta score for contrastive text analysis.**

## What is the purpose?

This script implements a relatively simple measure for distinctive features in two groups of texts. It allows you to find out which words are characteristic of one group of texts when compared to another group of texts.

The underlying measure of distinctiveness or keyness has been proposed by John Burrows under the name of `Zeta`hence the name of this Python package. 

## Getting help

* The `sampledata` folder contains some examples of what input pyzeta needs and what output it produces
* The `howto.md` file contains a brief tutorial of sorts for running analyses with pyzeta.

## Requirements and installation

* Requirements: Python 3 with pandas, numpy, sklearn, pygal, treetaggerwrapper (and TreeTagger)
