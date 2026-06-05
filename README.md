# Disentangling conviction and conformity: a Bayesian ideal point model of voting behaviour in online debates

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2606.03786)

Official repository for the paper:

> **Disentangling conviction and conformity: a Bayesian ideal point model of voting behaviour in online debates**  
> Elena Candellone

## Overview

This repository contains the code, data processing scripts, and analysis pipeline used in the paper.

### Abstract

Online debate platforms offer a unique window into the mechanisms driving opinion formation: they capture
both explicit political preferences and the peer environment in which those preferences are expressed. In this
work, I develop a Bayesian logistic regression model, inspired by ideal point models from political science,
to disentangle two competing mechanisms of voting behaviour in online debates: conviction, driven by prior
ideological beliefs, and conformity, driven by peer influence. I apply this framework to the Debate.org dataset,
comprising approximately 341k votes across 78k debates on 48 socio-political topics. As the debate platform does
not provide predefined topic labels for each debate, I infer the topic and stance from the debate text using large
language models, and, with a Bayesian approach, I quantify the relative contribution of each mechanism. I find
substantial heterogeneity across topics: conviction dominates on issues tied to personal freedoms and lifestyle
choices, such as drug legalisation and legalised prostitution, while conformity dominates on several topics widely
regarded as paradigmatic cases of moral conviction, including abortion, gun rights, and global warming. These
results have implications for the stability of online political discourse and the design of deliberative platforms.

## Repository Structure

```

project/
├── scripts/                           # Analysis scripts
    ├── 1_data_cleaning.py             # Data processing from raw data
    ├── 2x_debate_classification_x.py  # Classification of debates with gpt or bert
    ├── 3_user_vectors.py              # Generate user vectors from self-reported beliefs
    ├── 4x_debate_vectors_x.py         # Generate debate vectors from classification (gpt or bert)
    ├── 5x_run_model_x.py              # Run Bayesian model with Stan (gpt or bert)
    └── model.stan                     # Stan code containing the model
├── src/                               # Imports and additional scripts
├── results/                           # Generated results
├── plots/                             # Figures used in the manuscript
├── ddo.yml                            # Conda environment
└── README.md

````

## Installation

### Clone the repository

```bash
git clone https://github.com/elenacandellone/DDO-conviction-conformity.git
cd DDO-conviction-conformity
````

### Create environment

Using Conda:

```bash
conda create --name ddo --file ddo.yaml
conda activate ddo
```

## Data

### Access

Describe where the data comes from.

* Public dataset: [Dataset Name](dataset_link)
* Processed data used in the paper: `data/processed/`

### Reproducing preprocessing

```bash
python scripts/1_data_cleaning.py
```

## Reproducing Results

### Main experiments

```bash
python scripts/5a_run_model_gpt.py
```

### Generate figures

```bash
b_figures.ipynb
```


## Citation

If you use this code/paper, please cite:

```bibtex
@misc{candellone2026disentangling,
    title={Disentangling conviction and conformity: a Bayesian ideal point model of voting behaviour in online debates}, 
    author={Elena Candellone},
    year={2026},
    eprint={2606.03786},
    archivePrefix={arXiv},
    primaryClass={physics.soc-ph},
    url={https://arxiv.org/abs/2606.03786}, 
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

For questions regarding the code or paper:

* Elena Candellone
* [e.candellone@uu.nl](mailto:e.candellone@uu.nl)
