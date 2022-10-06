

## Purpose

This repo contains the scripts used in the following study:

`Single nucleotide variants in Pseudomonas aeruginosa populations from sputum correlate with baseline lung function and predict disease progression in individuals with cystic fibrosis`

---

## Project structure
```
├── CF_environment.yml
├── README.md
├── scripts
│   ├── Clustermap_plotting.py
│   ├── Data_preprocessing.py
│   ├── Machine_learning_analysis.py
│   └── SNV_filtering.py
```

---

## Content

All the scripts are tested on `Ubuntu 20.04 LTS` Linux operating system.

* `CF_environment.yml` : Script depedencies, to install all the dependencies, run the following command:
    `conda\mamba env create -f CF_environment.yml`

* `scripts\Clustermap_plotting.py` : Python script to plot clusetermap
* `scripts\Data_preprocessing.py` : Python script to pre-process genomic and meta-data
* `scripts\Machine_learning_analysis.py` : Python script to run Machine learning models and plot ROCAUCs
* `scripts\SNV_filtering.py` : Python script to filter out non-synonymous variants