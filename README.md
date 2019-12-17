# BERT based PAS analysis

## Set Up Environment

1. clone this repository
2. `pipenv sync`
3. `pipenv shell`

## Preprocess Documents

First, you need to load corpus and pickle it.

```zsh
python src/preprocess.py --kwdlc <path-to-KWDLC-directory> --kc <path-to-KyotoCorpus-directory> --out <path-to-output-directory>
```

## Configuring Settings

Before starting model training, prepare the configuration files.
The resultant files will be located at `config/`.

```zsh
python src/configure.py -c config -d <path-to-dataset-directory> -e <num-epochs> -b <batch-size> --model <model-name> --corpus kwdlc
```

example:

```zsh
python src/configure.py -c config -e 4 8 -b 8 --model BaselineModel --corpus kwdlc all
```

## Training Models

Launch the trainer with a configuration.

```zsh
python src/train.py -c <path-to-config-file> -d <gpu-ids>
```

example:

```zsh
python src/train.py -c config/BaselineModel-kwdlc-4e.json -d 0,1
```

## Testing Models

```zsh
python src/test.py -r <path-to-trained-model> -d <gpu-ids>
```

If you specify the config file, the setting is overwritten.

## Scoring From System Output

```zsh
python src/scorer.py --prediction-dir <system-output-directory> --gold-dir <gold-directory> --read-prediction-from-pas-tag
```

## Analyze Your Own data

```zsh
python src/inference.py -r <path-to-trained-model> --input "太郎はパンを買って食べた。" -tab -d <gpu-ids>
```

## Dataset

- /mnt/hinoki/ueda/kwdlc/new
- /mnt/hinoki/ueda/kc/split
