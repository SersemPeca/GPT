# Generative NMT (BG -> EN)

This project trains and runs a **generative neural machine translation** model built with PyTorch.

It uses a single autoregressive transformer that sees:
- source sentence tokens
- a separator token `<TRANS>`
- target sentence tokens

The default dataset setup is **Bulgarian source -> English target** (`train.bg` -> `train.en`).

## What This Project Does

- Prepares parallel corpora and builds a vocabulary
- Trains a transformer language model for translation
- Continues training from a saved checkpoint
- Translates new source sentences
- Evaluates with perplexity and corpus BLEU

## Project Files

- `run.py`: CLI entry point (`prepare`, `train`, `translate`, `perplexity`, `bleu`, etc.)
- `model.py`: transformer model and decoding (sampling + beam search)
- `utils.py`: corpus loading/tokenization and progress bar
- `parameters.py`: data paths and training/model hyperparameters
- `flake.nix`: Nix dev shell with Python + PyTorch + NLTK tooling

## Data Format

Expected files (see `parameters.py`):
- `en_bg_data/train.bg`
- `en_bg_data/train.en`
- `en_bg_data/dev.bg`
- `en_bg_data/dev.en`

Format rules:
- One sentence per line
- Parallel files must be line-aligned
- Sentences are tokenized with `nltk.word_tokenize`

## Environment Setup

### Option 1: Nix (recommended in this repo)

```bash
nix develop
```

This shell provides torch/numpy/nltk and sets `NLTK_DATA`.

### Option 2: Python virtualenv

Install at minimum:
- `torch`
- `numpy`
- `nltk`

Also ensure NLTK punkt resources are available (the code calls `nltk.download('punkt')`).

## How To Use

Run commands from the project root.

1. Prepare data and vocabulary:
```bash
python run.py prepare
```

2. Train:
```bash
python run.py train
```

3. Resume training from saved checkpoint:
```bash
python run.py extratrain
```

4. Translate a source file:
```bash
python run.py translate en_bg_data/dev.bg predictions.en
```

5. Evaluate perplexity on parallel files:
```bash
python run.py perplexity en_bg_data/dev.bg en_bg_data/dev.en
```

6. Evaluate BLEU (reference vs hypothesis files):
```bash
python run.py bleu en_bg_data/dev.en predictions.en
```

7. Generate continuation from a raw token prefix (debug utility):
```bash
python run.py generate "<S> example tokens <TRANS>"
```

## Artifacts Created

- `corpusData`: pickled train/dev integerized corpora
- `wordsData`: pickled `word -> index` vocabulary
- `NMTmodel`: saved model weights
- `NMTmodel.optim`: optimizer/checkpoint state

## Configuration

Edit `parameters.py` to change:
- dataset paths
- model size (`d_model`, `num_layers`, `num_heads`)
- optimization settings (`learning_rate`, `batch_size`, `max_epochs`, etc.)
