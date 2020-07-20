# BERT based PAS analysis

Japanese Predicate Argument Structure (PAS) Analyzer using BERT.

## Description

This project is mainly for Japanese PAS analysis,
but provides following other analyses as well.

- Coreference resolution
- Bridging anaphora resolution
- Nominal predicate argument structure analysis

PAS analysis process is as follows:

1. Apply Juman++ and (BERT)KNP to input text and split into base phrases
2. Extract predicates from base phrases by seeing whether the phrase has "<用言>" feature
3. Split each phrases into subwords using BPE
4. For each predicate subword, select its arguments

## Demo

<http://lotus.kuee.kyoto-u.ac.jp/~ueda/demo/bert-pas-analysis-demo/index.cgi>

<img width="1440" alt="スクリーンショット 2020-01-30 22 43 56" src="https://user-images.githubusercontent.com/25974220/73454841-3f180580-43b2-11ea-8251-f7e90e6db743.png">

## Requirements

- Python 3.7.2+
- [Juman++](https://github.com/ku-nlp/jumanpp) 2.0.0-rc3
- [KNP](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP) 4.2
- BERTKNP (optional)

## Setup Environment

Use [poetry](https://github.com/python-poetry/poetry)

`poetry install`

## Quick Start

```zsh
MODEL=/mnt/elm/ueda/bpa/result/best/model_best.pth

python src/predict.py \
--model $MODEL \
--input "太郎はパンを買って食べた。"
```

Result:

```text
太郎Cはp──┐
  パンnをp┐│
     買ってv┤  太郎:ガ パン:ヲ :ニ :ガ２ :ノ
    食べたv。*  太郎:ガ パン:ヲ :ニ :ガ２ :ノ
```

Options:

- `--model, -m, -r`: path to trained checkpoint
- `--device, -d`: GPU IDs separated by "," (if not specified, use CPU)
- `--input, -i`: input sentence or document separated by "。"
- `-tab`: output is KNP tab format
- `--use-bertknp`: use BERTKNP instead of KNP (requires BERTKNP)

## Analyze a Large Number of Documents

Given raw sentences, first you need to apply (BERT)KNP to them and split into documents.
Then, run `predict.py` specifying the document directory.

```zsh
python src/predict.py \
--model /mnt/elm/ueda/bpa/result/best/model_best.pth \
--knp-dir <path-to-parsed-document-directory>
--export-dir <path-to-export-directory>
```

For details, see Makefile [here](https://bitbucket.org/ku_nlp/causal-graph/src/master/scripts/knp_and_pas/Makefile)

## Training Your Own Model

### Preparing Corpus

```zsh
cd /somewhere
mkdir kwdlc kc
```

download corpora:

for menbers of bitbucket.org:ku_nlp
- `git clone https://github.com/ku-nlp/KWDLC kwdlc/KWDLC`
- `git clone git@bitbucket.org:ku_nlp/kyotocorpus.git kc/kyotocorpus`

otherwise
- `git clone https://github.com/ku-nlp/KWDLC kwdlc/KWDLC`
- `git clone https://github.com/ku-nlp/KyotoCorpus kc/KyotoCorpus`
- follow [instructions of KyotoCorpus](https://github.com/ku-nlp/KyotoCorpus#conversion-to-the-complete-annotated-corpus)

add features:

[kyoto-reader](https://github.com/ku-nlp/kyoto-reader) provides [some commands](https://kyoto-reader.readthedocs.io/en/latest/#corpus-preprocessor) to preprocess corpus.
Make sure you are in the virtual environment of bert_pas_analysis when you run `configure` and `idsplit` commands.

```
$ git clone https://github.com/ku-nlp/JumanDIC
$ configure --corpus-dir /somewhere/kwdlc/KWDLC/knp \
--data-dir /somewhere/kwdlc \
--juman-dic-dir /somewhere/JumanDIC/dic
created Makefile at /somewhere/kwdlc
$ configure --corpus-dir /somewhere/kc/kyotocorpus/knp \
--data-dir /somewhere/kc \
--juman-dic-dir /somewhere/JumanDIC/dic
created Makefile at /somewhere/kc
$ cd /somewhere/kwdlc && make -i
$ cd /somewhere/kc && make -i
$ idsplit --corpus-dir /somewhere/kwdlc/knp \
--output-dir /somewhere/kwdlc \
--train /somewhere/kwdlc/KWDLC/id/split_for_pas/train.id \
--valid /somewhere/kwdlc/KWDLC/id/split_for_pas/dev.id \
--test /somewhere/kwdlc/KWDLC/id/split_for_pas/test.id
$ idsplit --corpus-dir /somewhere/kc/knp \
--output-dir /somewhere/kc \
--train /somewhere/kc/kyotocorpus/id/split_for_pas/train.full.id \
--valid /somewhere/kc/kyotocorpus/id/split_for_pas/dev.full.id \
--test /somewhere/kc/kyotocorpus/id/split_for_pas/test.full.id
```

### Preprocessing Documents

After preparing corpora, you need to load and pickle them.

```zsh
python src/preprocess.py \
--kwdlc /somewhere/kwdlc \
--kc /somewhere/kc \
--out /somewhere/dataset \
--bert-name nict
--bert-path /somewhere/NICT_BERT-base_JapaneseWikipedia_32K_BPE
```

Don't care if many "sentence not found" messages are shown when processing kc.
It is a natural result of splitting document.

### Configuring Settings

Before starting model training, prepare the configuration files.
The resultant files will be located at `config/`.

```zsh
python src/configure.py \
-c <path-to-config-directory> \
-d /somewhere/dataset \
-e <num-epochs> \
-b <batch-size> \
--model <model-name> \
--corpus all
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
python src/train.py -c config/BaselineModel-all-4e-nict-cz-vpa.json -d 0,1
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

Here is an example of training your own model for 5 times with different random seeds:

```zsh
make train GPUS=<gpu-ids> CONFIG=<path-to-config-file> TRAIN_NUM=5
```

Testing command is as follows (outputs confidence interval):

```zsh
make test GPUS=<gpu-ids> RESULT=<path-to-result-dir>
```

This command executes above two commands all at once.

```zsh
make all GPUS=<gpu-ids> CONFIG=<path-to-config-file> TRAIN_NUM=5
```

Ensemble test is also available.

```zsh
make test-ens GPUS=<gpu-ids> RESULT=<path-to-result-dir>
```

## Environment Variables

- `BPA_CACHE_DIR`: A directory where processed document is cached. Default is `/data/$USER/bpa_cache`.
- `BPA_OVERWRITE_CACHE`: If set, bert_pas_analysis doesn't load cache even if it exists.
- `BPA_DISABLE_CACHE`: If set, bert_pas_analysis doesn't load or save cache.

## Dataset

- Kyoto University Web Document Leads Corpus ([KWDLC](https://github.com/ku-nlp/KWDLC))
- Kyoto University Text Corpus ([KyotoCorpus](https://github.com/ku-nlp/KyotoCorpus))

## Licence

MIT

## Author

Nobuhiro Ueda <ueda **at** nlp.ist.i.kyoto-u.ac.jp>
