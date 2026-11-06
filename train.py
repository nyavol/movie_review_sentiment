
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from util import Vocab, load_splits, make_loader
from models import DEFAULT_CONFIGS, build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gold = [], []
    for ids, lengths, y in loader:
        logits = model(ids.to(device), lengths.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(y.tolist())
    return accuracy_score(gold, preds), f1_score(gold, preds, average="macro"), preds, gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="lstm: 1e-3,transformer: 5e-4")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=256, help="tokens kept per text")
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    lr = args.lr or (1e-3 if args.model == "lstm" else 5e-4)

    train, val, test, label_names = load_splits(seed=args.seed)
    vocab = Vocab.build([t for t, _ in train], max_size=args.vocab_size)

    train_dl = make_loader(train, vocab, args.max_len, args.batch_size, shuffle=True)
    val_dl = make_loader(val, vocab, args.max_len, args.batch_size * 2, shuffle=False)
    test_dl = make_loader(test, vocab, args.max_len, args.batch_size * 2, shuffle=False)

    cfg = dict(DEFAULT_CONFIGS[args.model])
    if args.model == "transformer":
        cfg["max_len"] = max(cfg["max_len"], args.max_len)
    model = build_model(args.model, len(vocab), len(label_names), **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    warmup = max(1, int(0.06 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / (total_steps - warmup)),
    )
    criterion = nn.CrossEntropyLoss()
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{args.model}_imdb.pt")
    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, correct, seen = time.time(), 0.0, 0, 0
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for ids, lengths, y in bar:
            ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
            logits = model(ids, lengths)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += y.size(0)
            bar.set_postfix(loss=f"{running / seen:.4f}", acc=f"{correct / seen:.4f}")

        val_acc, val_f1, _, _ = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: train_loss={running / seen:.4f} train_acc={correct / seen:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({time.time() - t0:.0f}s)") #used chatgpt to make this line

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1, preds, gold = evaluate(model, test_dl, device)
    print(f"{args.model} test accuracy = {test_acc:.4f}  (macro-F1 {test_f1:.4f})")
    print(classification_report(gold, preds, target_names=label_names, digits=4))

    torch.save(
        {
            "model_type": args.model,
            "model_config": cfg,
            "state_dict": best_state,
            "vocab": vocab.word_list,
            "labels": label_names,
            "max_len": args.max_len,
            "dataset": "imdb",
            "val_acc": best_val,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
        },
        ckpt_path,
    )
    print(f"checkpoint saved in {ckpt_path}")

    results_path = os.path.join(out_dir, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    results[f"{args.model}_imdb"] = {
        "val_acc": round(best_val, 4), "test_acc": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4), "params": n_params, "epochs": args.epochs,
    }
    json.dump(results, open(results_path, "w"), indent=2)
    print(f"result in {results_path}")


if __name__ == "__main__":
    main()