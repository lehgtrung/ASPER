
## This is the code for the paper
**ASPER: Answer Set Programming Enhanced Neural Network Models 
for Joint Entity-Relation Extraction** [pdf](https://arxiv.org/abs/2305.15374)

The code is based on the following paper's code:
**SpERT: Span-based Entity and Relation Transformer**
[link to github code](https://github.com/lavis-nlp/spert)

## Setup (copy from SpERT's)
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
  - clingo (version > 5.5), [installation instruction](https://potassco.org/clingo/)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### Fetch data
Fetch converted (to specific JSON format) CoNLL04 \[1\] (we use the same split as \[4\]), SciERC \[2\] and ADE \[3\] datasets (see referenced papers for the original datasets):
```
bash ./scripts/fetch_datasets.sh
```

## Run the code

### Split data
```bash
python split_data.py <dataset> <percentage> <num_folds>
```
For example: we want split CoNLL04 dataset into 5 folds with 20% labeled data
```bash
python split_data.py conll04 20 5
```

### To run ASPER
```bash
python run_ker.py --dataset <dataset> --fold <fold_number> --percent <percentage> --max_iter <max_iteration>
```
For example: we want to run on CoNLL04 dataset with fold #1, with 20% labeled data, maximum 5 iterations 
```bash
python run_ker.py --dataset conll04 --fold 1 --percent 20 --max_iter 5
```

### To run comparison methods
The command for running comparison methods (self/curriculum/tri-training) is similar,
For example: for comparison, we want to run tri-training with fold #1, with 20% labeled data, maximum 5 iterations
```bash
python run_tri_training.py --dataset conll04 --fold 1 --percent 20 --max_iter 5
```