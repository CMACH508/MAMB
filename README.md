# MAMB

This repository contains the implementation code, trained model and part of the test set for our work: A Deep Learning Method with Multi-view Attention and Multi-branch GCN for BECT Diagnosis. 



## Introduction

Epilepsy is a prevalent chronic neurological disorder in childhood, imposing a heavy burden on patients and their families. The development of deep learning and the accumulation of clinical medical data has led to a surge in neural network algorithms proposed for the automatic detection of childhood epilepsy using electroencephalogram (EEG) signals. However, for Benign Childhood Epilepsy with Centro-Temporal Spikes (BECT), the most common type of childhood epilepsy, there is no large-scale public dataset available for its detection. Moreover, although there were a few studies of BECT using private data, they focused only on identifying the presence of abnormal discharges, ignoring the sleep stage of the discharges, which is actually a critical factor for the doctor's diagnosis. To tackle these challenges, we create a BECT dataset containing 38K real samples from 100 subjects, meticulously annotated by neurology experts. Building on this dataset, we propose a multi-view attention and multi-branch graph convolutional neural network (MAMB) to differentiate abnormal discharges and classify sleep stages in the samples. Experimental results demonstrate that the model benefits from three-dimensional attention mechanisms in the spatial, temporal, and spectral domains, allowing better exploration of intrinsic relationships within EEG signals. Additionally, the sleep and BECT branches enhance the model's ability to detect abnormal discharges and differentiate various sleep stages. Our model achieves 88.07% and 90.91% accuracy for BECT classification and sleep stage staging, respectively, and 81.93% for the four-classification task of simultaneously judging BECT and sleep stage. In addition, the introduction of the multi-view attention mechanism makes the model interpretable and raises hopes of further assisting experts in disease and medication analysis.



## Overview

<img src=".\overview.png" width="100%" />



## Requirements

python==3.6.13

numpy==1.19.5

scikit_learn==0.24.2

torch==1.10.1


Install dependencies:

```
pip install -r requirements.txt
```


## Test

`models.py` contains the MAMB model we proposed.


For the protection of patient privacy and in accordance with the hospital's permission, we are currently providing randomly generated samples only to demonstrate the feasibility of our model, and they do not represent real performance.

To test MAMB, run

```
python test.py
```


## Pre-trained Models

`pretrained_model.pt` is the trained model of MAMB.