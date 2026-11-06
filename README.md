# LSTM vs Transformer — IMDB Sentiment Classification

Two text classifiers built from scratch in PyTorch and compared on the [IMDB movie review dataset](https://huggingface.co/datasets/stanfordnlp/imdb) (binary sentiment: positive / negative):

- **BiLSTM**: a bidirectional LSTM with packed sequences and concatenated mean + max pooling over the hidden states.
- **Transformer encoder**: written by hand (no `nn.Transformer`). It has multi-head self-attention, sinusoidal positional encoding, pre-LayerNorm blocks, a GELU feed-forward layer, padding masks and masked mean pooling.

Both models have about 4.1M parameters, so the comparison is fair.

## Results

| Model       | Params | Epochs | Val acc | Test acc | Test macro-F1 |
|-------------|-------:|-------:|--------:|---------:|--------------:|
| BiLSTM      | 4.17M  | 40     | 0.8914  | **0.8693** | **0.8693** |
| Transformer | 4.11M  | 7      | 0.8544  | 0.8432   | 0.8432        |

The test set is the official 25k IMDB test split. Validation is a stratified 20% holdout from the train split.

## Project structure

```
├── util.py         # tokenizer, vocabulary, dataset splits, DataLoader/collate
├── models.py       # LSTMClassifier, TransformerClassifier (custom attention), configs
├── train.py        # training loop: AdamW, linear warmup/decay, grad clipping, best-val checkpointing
├── inference.py    # SentimentPredictor class + CLI (single text or interactive)
└── checkpoints/    # created by train.py (weights + results.json), not tracked in git
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --model lstm
python train.py --model transformer --epochs 7
```

Useful flags: `--epochs`, `--batch-size`, `--lr` (defaults: 1e-3 for the LSTM, 5e-4 for the Transformer), `--max-len` (default 256 tokens), `--vocab-size`, `--seed`.

## Inference

Train a model first. Training writes its checkpoint to `checkpoints/<model>/`.

```bash
# single prediction
python inference.py --checkpoint checkpoints/lstm/lstm_imdb.pt --text "An absolute masterpiece, I loved every minute."

# interactive mode (press Enter on an empty line to quit)
python inference.py --checkpoint checkpoints/transformer/transformer_imdb.pt
```

Example output:

```
 Text: "An absolute masterpiece, I loved every minute."
  [lstm] -> POSITIVE
      positive   1.0000  ##############################
      negative   0.0000
```

From Python:

```python
from inference import SentimentPredictor

p = SentimentPredictor("checkpoints/lstm/lstm_imdb.pt")
p.predict("Boring and way too long.")
# {'label': 'negative', 'scores': {'negative': ..., 'positive': ...}}
```
