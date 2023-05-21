# eyos_assessment

## Problem Statement

Within each name group (barcodes), there might be outliers where the product name looks clean, but intuitively differs from other names in the group.

Example given:
* SEAGULL NATPH 25g WRNA / PCS
* SEA GULL WARNA RENTENG
* MANGKOK SAMBALL ALL VAR
* SEAQULL NAPT WARNA 25GR
* \- SEA GULL NAPHT 25GR SG-519W 1PCSX 1.500,00:

I am to formulate a solution using Data Science techniques to create a solution to this problem.

## Assessment Plan

The product names are given as a string, where there might include many redundant information, such as quantity or single letters etc.

1. Remove unnecessary information from product name using Named Entity Recognition: 
    * QTY (quantity)
    * PROD (product name)
    * ADJ (adjective/descriptors)
    * O (not important words)
2. Use ensemble of embeddings to convert strings/texts into numbers
    * TF-IDF
    * Bert embeddings
3. Similarity score on the embeddings using various distance functions
4. Outlier detection using weighted average of the features

## Directory Structure

* `eyos_assessment.ipynb` - notebook documenting the whole process of the project
* `train_script.py` - training and inference script for Bert NER