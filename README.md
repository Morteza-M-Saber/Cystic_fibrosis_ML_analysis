

## Purpose

This repo contains the scripts written for the analysis in the following study:

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

All the scripts are tested on `Ubuntu 20.04 LTS` Linux operating system with `gcc compiler 5.4.0`.

* `CF_environment.yml` : Script depedencies, to install all the dependencies

* `scripts\Clustermap_plotting.py` : Python script to plot clusetermap.
* `scripts\Data_preprocessing.py` : Python script to pre-process genomic and meta-data.
* `scripts\Machine_learning_analysis.py` : Python script to train Machine learning models, evaluate the results and plot ROCAUCs.
* `scripts\SNV_filtering.py` : Python script to filter out non-desired variants.


## Raw sequencing data to ML analysis guide

Required dependencies should be installed as listed in the `CF_environment.yml` file by:

`conda create -f CF_environment.yml`



### 1. Trimming raw sequencing reads

Raw sequencing reads were trimmed based on per-base phred quality score cutoff (‘q’ flag) of 18, window size of 1 base pair and minimum remaining sequence length (‘l’ flag) of 19 using fastq-mcf (v.1.04.636) (Aronesty 2013). 

### 2. Mapping reads to PES genome
Reads were aligned to the PES genome (CP080405) using BWA MEM (Li, 2013), and the alignments were sorted and indexed using SAMtools (v.1.9) (Li, Handsaker et al. 2009). Samples with average sequencing depth <= 10X across the target genes were discarded.

### 3. Calling single nucleotide variants (SNV)
Single-nucleotide variants (SNVs) with minimum mapping quality of 20, minimum base quality of 18 and minimum coverage of 10x were then identified using VarScan 2 (Koboldt, Zhang et al. 2012)

### 4. Functional annotation of SNVs
Functional consequences of each SNV were inferred using snpEFF (v.2.4.2) (Cingolani, Platts et al. 2012).The SNV allele frequencies (ranging from 0 to 1) at each polymorphic site covered by the AmpliSeq panel were used to generate a SNV frequency matrix, with samples as rows and nucleotide positions as columns.

### 5. Filtering out synonymous SNVs
Synonymous SNVs (i.e SNVs with no effect on amino acid composition of protein) were filtered out using the script
>python3 scripts\SNV_filtering.py

### 6. Merging genomic and clinical meta-data to generate ML input matrix

Non-synonymous and clinical meta-data were merged to generate ML input matrix.

Script:
>python3 scripts\Data_preprocessing.py

ML input table:
>intermediate_files\CF_NSV_maf0.01_with_metadata.matrix.pickle

### 7. Train and evaluate performance of Machine learning models

To train/evalaute machine learning models

   > python3 scripts\Machine_learning_analysis.py