"""
predict_star.py
~~~~~~~~~~~~~~~~
Load a fine‑tuned STAR classifier (produced by `star_trainer.py`) and predict
labels for new sentences.

Features
--------
* Accepts input as a **text file** (one sentence per line) *or* direct
  command‑line arguments.
* Prints a nicely formatted table to stdout **and/or** writes a CSV/JSONL file.
* Returns the top label per sentence **plus** class probabilities.

Usage examples
--------------
# Predict sentences passed inline
python predict_star.py \
  --model_dir star_output/best_model \
  --sentences "I organised a cross‑team retrospective." \
             "We reduced downtime by 40 percent." 

# Batch mode from file and save CSV
python predict_star.py \
  --model_dir star_output/best_model \
  --input_file examples.txt \
  --output_csv predictions.csv

Dependencies
------------
 pip install transformers torch pandas tabulate
"""

import argparse, sys, pathlib, json
import torch, pandas as pd
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tabulate import tabulate

LABEL_MAPPING = {
    "LABEL_0": "Action",
    "LABEL_1": "Result",
    "LABEL_2": "Situation",
    "LABEL_3": "Task",
}


def parse_args():
    p = argparse.ArgumentParser(description="STAR sentence predictor")
    p.add_argument(
        "--model_dir",
        required=True,
        help="Path or HF repo id of the fine‑tuned model directory",
    )

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input_file", help="Text file, one sentence per line")
    grp.add_argument("--sentences", nargs="+", help="Sentence(s) to classify")

    p.add_argument("--output_csv", help="Optional CSV to save predictions")
    p.add_argument("--output_jsonl", help="Optional JSONL to save predictions")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to run inference (auto picks CUDA if available)",
    )
    return p.parse_args()


def load_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def predict(sentences, tokenizer, model, device, max_len):
    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
        probs = softmax(logits, dim=-1).cpu().numpy()
    labels = model.config.id2label
    preds = probs.argmax(axis=-1)
    results = []
    for sent, idx, prob_vec in zip(sentences, preds, probs):
        label = labels[int(idx)] if isinstance(labels, dict) else idx
        results.append(
            {
                "sentence": sent,
                "label": LABEL_MAPPING[label],
                **{
                    f"prob_{labels[i] if isinstance(labels, dict) else i}": float(p)
                    for i, p in enumerate(prob_vec)
                },
            }
        )
    return results


def main():
    args = parse_args()

    # 1. Collect sentences
    if args.input_file:
        txt = pathlib.Path(args.input_file).read_text(encoding="utf‑8")
        sentences = [l.strip() for l in txt.splitlines() if l.strip()]
    else:
        sentences = args.sentences
    if not sentences:
        sys.exit("[ERROR] No sentences to classify.")

    # 2. Load model/tokenizer
    tokenizer, model, device = load_model(args.model_dir, args.device)

    # 3. Predict
    results = predict(sentences, tokenizer, model, device, args.max_length)

    # 4. Display pretty table
    df = pd.DataFrame(results)
    print(tabulate(df[["sentence", "label"]], headers="keys", showindex=False))

    # 5. Optional exports
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Saved CSV to {args.output_csv}")
    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf‑8") as f:
            for row in df.to_dict(orient="records"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"✓ Saved JSONL to {args.output_jsonl}")


if __name__ == "__main__":
    main()
