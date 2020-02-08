# BERT based PAS analysis

Japanese Predicate Argument Structure (PAS) Analyzer using BERT.

## Description

This project is mainly for Japanese PAS analysis,
but provides following other analyses as well.

- coreference resolution
- bridging anaphora resolution
- eventive noun argument structure analysis

PAS analysis process is as follows:

1. apply Juman++ and (BERT)KNP to input text and split into tags
2. extract predicates from tags by seeing whether the tag has "<用言>" feature
3. split each tags into subwords using BPE
4. for each predicate subword, select its arguments

## Demo

<http://lotus.kuee.kyoto-u.ac.jp/~ueda/demo/bert-pas-analysis-demo/index.cgi>

<img width="1440" alt="スクリーンショット 2020-01-30 22 43 56" src="https://user-images.githubusercontent.com/25974220/73454841-3f180580-43b2-11ea-8251-f7e90e6db743.png">

## Requirements

- Python 3.6.5
- PyTorch 1.3.1
- Transformers 2.1.1
- pyknp 0.4.1
<!-- - kyoto-reader 0.0.1 -->

## Quick Start

```zsh
MODEL=/mnt/hinoki/ueda/bert/pas_analysis/result/best/model_best.pth

python src/inference.py \
--model $MODEL \
--input "太郎はパンを買って食べた。"
```

Result:

```text
太郎Cはp──┐
  パンnをp┐│
     買ってv┤  太郎:ガ パン:ヲ NULL:ニ NULL:ガ２ NULL:ノ
    食べたv。*  太郎:ガ パン:ヲ NULL:ニ NULL:ガ２ NULL:ノ
```

Options:

- `--model, -m, -r`: path to trained checkpoint
- `--device, -d`: GPU IDs separated by "," (if not specified, use CPU)
- `--input, -i`: input sentence or document separated by "。"
- `-tab`: output is KNP tab format
- `--use-bertknp`: use BERTKNP instead of KNP

## Analyze a Large Number of Documents

Given raw sentences, first you need to apply (BERT)KNP to them
and split into documents.
Next, specify the document directory when you run `inference.py`

```zsh
python src/inference.py \
--model /mnt/hinoki/ueda/bert/pas_analysis/result/best/model_best.pth \
--knp-dir <path-to-parsed-document-directory>
--export-dir <path-to-export-directory>
```

For details, see Makefile [here](https://bitbucket.org/ku_nlp/causal-graph/src/master/scripts/knp_and_pas/Makefile)

## Training Your Own Model

### Preprocess Documents

First, you need to load corpus and pickle it.

```zsh
python src/preprocess.py \
--kwdlc <path-to-KWDLC-directory> \
--kc <path-to-KyotoCorpus-directory> \
--out <path-to-output-directory>
```

example:

```zsh
python src/preprocess.py \
--kwdlc /mnt/hinoki/ueda/kwdlc/new \
--kc /mnt/hinoki/ueda/kc/split \
--out data/dataset
```

### Configuring Settings

Before starting model training, prepare the configuration files.
The resultant files will be located at `config/`.

```zsh
python src/configure.py \
-c config \
-d <path-to-dataset-directory> \
-e <num-epochs> \
-b <batch-size> \
--model <model-name> \
--corpus kwdlc
```

example:

```zsh
python src/configure.py -c config -d data/dataset -e 4 8 -b 8 --model BaselineModel --corpus kwdlc all
```

### Training Models

Launch the trainer with a configuration.

```zsh
python src/train.py \
-c <path-to-config-file> \
-d <gpu-ids>
```

example:

```zsh
python src/train.py -c config/BaselineModel-kwdlc-4e.json -d 0,1
```

### Testing Models

```zsh
python src/test.py \
-r <path-to-trained-model> \
-d <gpu-ids>
```

If you specify the config file, the setting will be overwritten.

You can perform ensemble test as well:

```zsh
python src/test.py \
--ens <path-to-model-set-directory> \
-c <path-to-config-file> \
-d <gpu-ids>
```

### Scoring From System Output

```zsh
python src/scorer.py \
--prediction-dir <system-output-directory> \
--gold-dir <gold-directory> \
--read-prediction-from-pas-tag
```

## Perform Training Process with Make

You can also perform training and testing via make command.

Here is an example of training your own model for 10 times with different random seeds:

```zsh
make train GPUS=<gpu-ids> CONFIG=<path-to-config-file> TRAIN_NUM=10
```

Testing command is as follows (outputs confidence interval):

```zsh
make test GPUS=<gpu-ids> RESULT=<path-to-result-dir>
```

This command executes above two commands all at once.

```zsh
make all GPUS=<gpu-ids> CONFIG=<path-to-config-file> TRAIN_NUM=10
```

Ensemble test is also available.

```zsh
make test-ens GPUS=<gpu-ids> RESULT=<path-to-result-dir>
```

## Environment Variables

- `BPA_CACHE_DIR`: A directory where processed document is cached. Default is `/data/$USER/bpa_cache`.
- `BPA_OVERWRITE_CACHE`: If set, bert_pas_analysis doesn't load cache even if it exists.

## Dataset

- /mnt/hinoki/ueda/kwdlc/new
- /mnt/hinoki/ueda/kc/split

## Licence

MIT

## Author

Nobuhiro Ueda <ueda **at** nlp.ist.i.kyoto-u.ac.jp>
