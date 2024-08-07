# BERT4ETH 

This is the PyTorch implementation for the paper [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3543507.3583345). 
This repository is revised from [the official repository](https://github.com/Bayi-Hu/BERT4ETH_PyTorch).


## Getting Start

### Requirements

PyTorch > 1.12.0

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization(ENS)](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [De-anonymization(Tornado)](https://drive.google.com/file/d/1DMbPSZMSvTYMKUZg3oYKFrjPo2_jeeG4/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)


#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 

```sh
cd BERT4ETH_PyTorch/data; # Labels are already included
unzip ...;
``` 

### Pre-training


#### Step 1: Transaction Sequence Generation

```sh
cd pretrain;
python gen_seq.py --bizdate=bert4eth_exp
```


#### Step 2: Pre-train BERT4ETH 

```sh
python run_pretrain.py --bizdate="bert4eth_exp" \
                       --ckpt_dir="bert4eth_exp"
```

#### Step 3: Output Representation

```sh
python run_embed.py --bizdate="bert4eth_exp" \
                       --init_checkpoint="bert4eth_exp/xxx.pth"
```

### Evaluation 

#### Phishing Account Detection
```sh
cd ..
cd downstream
python phish_detection_mlp.py --input_dir="../outputs/xxx"
```
#### De-anonymization (ENS dataset)

```sh
python run_dean_ENS.py --metric=euclidean \
                       --init_checkpoint=bert4eth_exp/model_104000
```


### Fine-tuning for phishing account detection
```
python phish_finetune.py
```
#### Evaluation 
```
python finetune_test.py 
```

-----
## Q&A

If you have any questions, you can either open an issue or contact the original author (sihaohu@gatech.edu), and I will reply as soon as I see the issue or email.

