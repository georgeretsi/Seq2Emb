# From Seq2Seq to Handwritten Word Embeddings

This repository contains the code for generating word embeddings and reproducing the experiments of our work: G. Retsinas, G. Sfikas, C. Nikou & P. Maragos, "From Seq2Seq to Handwritten Word Embeddings", BMVC, 2021.

We provide PyTorch implementations of all the main components of the proposed DNN system that can generate discriminative word embeddings, ideal for Keyword Spotting. The core idea is to make use of the encoding output of a Sequence-to-Sequence architecture, trained for performing word recognition. 

Overview of overall system (two recognition branches: CTC & Seq2Seq):
![System Overview](paper_figs/htr_overview.png?raw=true "System Overview")

Architectural Details:
![Architecture](paper_figs/detailed_arch.png?raw=true "Architecture")

Overview of the different setups of the proposed system. Evaluation can be categorized to recognition and three spotting cases: Query-by-Example (QbE), Query-by-String (QbS) and QbS by force-alignment (FA).
![Functionality](paper_figs/bmvc-functionality.png?raw=true "Functionality")


-------------------------------------------------------------------------
### Getting Started

#### Installation

```{bash}
git clone https://github.com/georgeretsi/Seq2Emb
cd Seq2Emb
```

Install PyTorch and 1.7+ and the other dependencies (e.g., scikit-image), as stated in requirements.txt file.
For pip users, please type the command ``pip install -r requirements.txt``.

*Initially tested on PyTorch 1.7 and subsequently tested on 1.9 version*


#### Dataset Setup (IAM)

The provided code includes a loader for the IAM Handwriting Database (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). Registration is required for downloading (https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database).

In order to use the provided loader the IAM datasets should be structured as follows (and the path should be provided as argument):

```
IAMpath
│    
└───set_split
│   │   trainset.txt
│   │   testset.txt
│   │   ...
│   
└───ascii
│   │   words.txt
│   │   lines.txt
│   │   ...
│   
└───words
    │  ...
```

-------------------------------------------------------------------------

### Training and Configuration Options

Train the models with default configurations:
```
train_words.py --dataset IAM --dataset_path IAMpath --gpu_id 0
```

*The gpu_id argument corresponds to the ID of the GPU to be used (single-GPU implementation).*

Extra arguments related to the proposed work are:
- lowercase (bool) : use reduced lowercase character set (typically used in KWS)
- path_ctc (bool) : add the CTC branch
- path_s2s (bool) : add the Seq2Seq branch
- path_autoencoder (bool) : add the character encoder branch (forming an autoencoder)
- train_external_lm (bool) : use an external lexicon for training the autoencoder path
- binarize (bool) : binarize the word embedding
- feat_size (int) : define the size of word embeddings


General configuration variables, such as training options (initial learning rate, batch size etc.) and architecture options (depth and configuration of the DNN components), can be set through the file ``config.py``.

NOTE: *example.py contains an alternative way to train the system using the function evaluate_setting of train_words.py*

-------------------------------------------------------------------------
### Citation:

```
@inproceedings{retsinas2021from,
  title={From Seq2Seq to Handwritten Word Embeddings},
  author={Retsinas, George and Sfikas, Giorgos and Nikou, Christophoros and Maragos, Petros},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2021},
}
```
