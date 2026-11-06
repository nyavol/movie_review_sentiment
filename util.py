import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

HF_DATASET = "stanfordnlp/imdb"
LABELS = ["negative", "positive"]

TOKEN_RE = re.compile(r"[a-z0-9']+|[!?.,;:()]") #used chatgpt for this


def tokenize(text: str):
    text = text.lower().replace("<br />", " ")
    return TOKEN_RE.findall(text)

def to_list(d):
    return list(zip(d["text"], d["label"]))

class Vocab:
    def __init__(self, word_list):
        self.word_list = list(word_list)
        self.word_dict = {tok: i for i, tok in enumerate(self.word_list)}

    @classmethod
    def build(cls, texts, max_size=30000, min_freq=2):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        words = [w for w, f in counter.most_common(max_size - 2) if f >= min_freq]
        return cls([PAD, UNK] + words)

    def encode(self, text, max_len):
        ids = [self.word_dict.get(tok, UNK_ID) for tok in tokenize(text)][:max_len]
        return ids if ids else [UNK_ID]

    def __len__(self):
        return len(self.word_list)


def load_splits(val_fraction=0.2, seed=42):
    data = load_dataset(HF_DATASET)
    split = data["train"].train_test_split(test_size=val_fraction, seed=seed, stratify_by_column="label")
    train, val, test = split["train"], split["test"], data["test"]

    return to_list(train), to_list(val), to_list(test), LABELS


class TextDataset(Dataset):
    def __init__(self, samples, vocab, max_len):
        self.items = [(vocab.encode(t, max_len), y) for t, y in samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.full((len(seqs), int(lengths.max())), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, lengths, torch.tensor(labels, dtype=torch.long)


def make_loader(samples, vocab, max_len, batch_size, shuffle):
    return DataLoader(
        TextDataset(samples, vocab, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )