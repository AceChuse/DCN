# Fuzzy-Parameter-Mapping-Rules

This repo contains code accompaning the paper. It includes code for running the few-shot supervised learning domain experiments, including sinusoid regression, Omniglot classification, and MiniImagenet classification.

### Dependencies
This code requires the following:
* python 3.\*
* Pytorch 0.4.1+

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `pre_data/omniglot_resized.py` and `pre_data/mini_images.py` respectively.

### Sinusoid
To run the code, see the usage instructions at the top of Sine/sinusoid_e.py.

### Omniglot
Generate few-shot learning task.
```bash
$ cd Omni
$ python omniglot_make_task.py
```
To run the code, see examples in Omni/omni_caps.py and Omni/omni_fmeta.py.
Automated training and test fuzzycaps4
```bash
$ python omni_caps.py
```
Automated training and test fuzzymeta4
```bash
$ python omni_fmeta.py
```

### MiniImagenet
Generate few-shot learning task.
```bash
$ cd Mini
$ python miniimagenet_make_task.py
```
To run the code, see examples in Omni/mini_auto_1shot.py and Mini/mini_auto_5shot.py.
Automated training and test fuzzymeta on 1-shot task.
```bash
$ python mini_auto_1shot.py
```
Automated training and test fuzzymeta on 5-shot task.
```bash
$ python mini_auto_5shot.py
```
