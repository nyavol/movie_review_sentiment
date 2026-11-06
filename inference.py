import argparse

import torch

from util import Vocab
from models import build_model


class SentimentPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model_type = ckpt["model_type"]
        self.labels = ckpt["labels"]
        self.max_len = ckpt["max_len"]
        self.test_acc = ckpt.get("test_acc")
        self.vocab = Vocab(ckpt["vocab"])

        self.model = build_model(self.model_type, len(self.vocab), len(self.labels), **ckpt["model_config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_batch(self, texts):
        encoded = [self.vocab.encode(t, self.max_len) for t in texts]
        lengths = torch.tensor([len(e) for e in encoded])
        ids = torch.zeros(len(encoded), int(lengths.max()), dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = torch.tensor(e)

        probs = self.model(ids.to(self.device), lengths.to(self.device)).softmax(-1).cpu()
        results = []
        for row in probs:
            scores = {lab: round(float(p), 4) for lab, p in zip(self.labels, row)}
            results.append({"label": max(scores, key=scores.get), "scores": scores})
        return results

    def predict(self, text):
        return self.predict_batch([text])[0]


def show(name, result):
    print(f"  [{name}] -> {result['label'].upper()}")
    for lab, s in sorted(result["scores"].items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<10} {s:.4f}  {'#' * int(round(s * 30))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint")
    ap.add_argument("--text", default=None, help="text")
    args = ap.parse_args()

    predictor = SentimentPredictor(args.checkpoint)
    acc = f"{predictor.test_acc:.4f}" if predictor.test_acc is not None else "n/a"
    print(f"Model {predictor.model_type} is loaded from {args.checkpoint} (test acc {acc}, labels {predictor.labels})")

    def run(text):
        print(f'\n Text: "{text}"')
        show(predictor.model_type, predictor.predict(text))

    if args.text:
        run(args.text)
        return

    try:
        while True:
            text = input("\n> ").strip()
            if not text:
                break
            run(text)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()