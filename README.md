# BERT based PAS analysis

## Set Up Environment

1. clone this repository
2. `pipenv sync`
3. `pipenv shell`

## Configuring Settings

Before starting model training, prepare the configuration files.
The resultant files will be located at `config/`.

```zsh
python src/configure.py -c config -e <num-epochs> -b <batch-size> --model <model-name> --corpus kwdlc
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

## Dataset

- /mnt/hinoki/ueda/kwdlc/new
- /mnt/hinoki/ueda/kc/split
