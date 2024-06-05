# Annotating FrameNet via Structure-Conditioned Language Generation
This repository is the Python implementation of our paper:

Annotating FrameNet via Structure-Conditioned Language Generation

Xinyue Cui, Swabha Swayamdipta

The 62nd Annual Meeting of the Association for Computational Linguistics, 2024

## Installation
### Create python environment
```
conda create -n framenet python=3.10.11
conda activate framenet
```
### Install python dependencies
```
pip install -r requirements.txt
```

## Data
Request and download Framenet Dataset 1.7 from [Website](https://framenet.icsi.berkeley.edu/fndrupal/). Name the dataset folder `fndata-1.7` and place it at the same directory level as the Python scripts.

## Preprocessing
### Preprocess FrameNet data
```
python preprocess.py
```
### Create train/test split
```
python train_test_split.py
```

## Training & Evaluation
### Conditional generation
Train T5 model for conditionl generation and save generated data by T5 and GPT-4 models conditioned on different levels of semantic information:
```
python generation.py
```
### Filtering
Train SpanBERT model for FE type classification and use it to filter out generated FE spans with inconsistent FE types as the original:
```
python filter.py
```
### SRL Parsing
Train SpanBERT model for SRL parsing and evaluate performance trained on unaugmented data and augmented data:
```
python srl_parser.py
```
