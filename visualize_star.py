"""
Enhanced visualizations for STAR‑classifier presentation.

Added in this version
---------------------
1. Top‑attention bar chart now:
   • filters special tokens + English stop‑words
   • merges WordPiece sub‑tokens
2. Dropout vs eval heat‑maps plotted side‑by‑side
3. Optional single layer/head heat‑map via --layer --head flags
4. Attention‑roll‑out attribution plot + HTML highlighting

Dependencies
------------
pip install torch transformers pandas matplotlib nltk seaborn ipython
First run (to download NLTK stop‑words):
>>> import nltk; nltk.download('stopwords')
"""

import argparse, os, math, html, webbrowser, tempfile
from collections import defaultdict
import numpy as np
import torch, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


# ----------------- AdamW training curve ---------------------------------- #


def plot_training_curves(csv_path, out_path):
    hist = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(hist["step"], hist["train_loss"], label="train loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(hist["step"], hist["val_f1"], "--", label="val F1", color="tab:orange")
    ax2.set_ylabel("F1")
    fig.legend(loc="lower right")
    fig.suptitle("AdamW training dynamics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------- Attention helpers ------------------------------------- #


def aggregate_attention(attentions, layer=None, head=None, average=True):
    """
    attentions: tuple(num_layers) of (B, H, S, S)
    If layer/head provided return that matrix; else optionally average heads+layers.
    """
    att = torch.stack(attentions)  # (L, B, H, S, S)
    if layer is not None:
        att = att[layer]  # (B, H, S, S)
    else:
        att = att.mean(0)  # average layers -> (B, H, S, S)
    if head is not None:
        att = att[:, head]  # (B, S, S)
    elif average:
        att = att.mean(1)  # average heads -> (B, S, S)
    return att[0]  # batch dim


def plot_heatmap(attn, tok_labels, ax, title):
    sns.heatmap(
        attn,
        xticklabels=tok_labels,
        yticklabels=tok_labels,
        cmap="viridis",
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tok_labels, rotation=0, fontsize=7)
    ax.set_title(title, fontsize=10)


# ----------------- WordPiece → word merging ------------------------------ #


def merge_wordpieces(tok_labels, scores):
    """
    Merge 'stack', '##ing' -> 'stacking' and add their scores.
    Returns lists of merged tokens and aggregated scores.
    """
    merged, agg = [], []
    buf, buf_score = "", 0.0
    for tok, sc in zip(tok_labels, scores):
        if tok.startswith("##"):
            buf += tok[2:]
            buf_score += sc
        else:
            if buf:
                merged.append(buf)
                agg.append(buf_score)
            buf, buf_score = tok, sc
    if buf:
        merged.append(buf)
        agg.append(buf_score)
    return merged, agg


def top_attention_bar(attn_mat, tok_labels, out_path, top_k=10):
    cls_row = attn_mat[0, 1:].cpu().numpy()  # exclude [CLS]→[CLS]
    tokens = tok_labels[1:]  # align
    tokens, scores = merge_wordpieces(tokens, cls_row)
    clean = [
        (t, s)
        for t, s in zip(tokens, scores)
        if t.lower() not in STOP_WORDS and t not in SPECIAL_TOKENS
    ]
    clean.sort(key=lambda x: x[1], reverse=True)
    toks, scs = zip(*clean[:top_k])

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(toks)), scs)
    plt.xticks(range(len(toks)), toks, rotation=45, ha="right")
    plt.ylabel("attention weight")
    plt.title(f"Top‑{top_k} tokens by [CLS] attention")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------- Attention roll‑out ------------------------------------ #


def attention_rollout(all_attn, add_residual=True):
    """
    Basic rollout (BERT‑Viz style):
      start from identity, multiply (A + I)/2 across layers.
    all_attn: tensor (L, B, H, S, S)
    returns (S,) importance wrt [CLS] token.
    """
    attn = torch.stack(all_attn)  # (L,B,H,S,S)
    if add_residual:
        resid = torch.eye(attn.size(-1)).to(attn.device)
        attn = attn + resid
        attn = attn / attn.sum(-1, keepdim=True)
    joint = attn.mean(2)  # avg heads (L,B,S,S)
    rollout = joint[0]  # (L,S,S)
    probs = rollout[0]
    for l in range(1, rollout.size(0)):
        probs = torch.matmul(rollout[l], probs)
    return probs[0]  # importance of each token wrt CLS


def plot_rollout_importance(imp, tok_labels, out_path):
    imp = imp.cpu().numpy()
    tokens, scores = merge_wordpieces(tok_labels, imp)
    tokens, scores = tokens[1:], scores[1:]  # drop [CLS]
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(tokens)), scores)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.title("Attention‑roll‑out token importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return tokens, scores


def html_highlight(tokens, scores, outfile):
    norm = np.array(scores)
    norm = (norm - norm.min()) / (np.ptp(norm) + 1e-9)
    colors = (255 * (1 - norm)).astype(int)  # white→blue
    spans = [
        f'<span style="background:rgb(0,0,{c});color:white"> {html.escape(t)} </span>'
        for t, c in zip(tokens, colors)
    ]
    html_str = "<p>" + "".join(spans) + "</p>"
    with open(outfile, "w") as f:
        f.write(html_str)
    return outfile


# ----------------- Main -------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sample_text", required=True)
    ap.add_argument("--train_log")
    ap.add_argument("--output_dir", default="./figures")
    ap.add_argument("--layer", type=int, help="single layer heat‑map (0‑based)")
    ap.add_argument("--head", type=int, help="single head heat‑map (0‑based)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, output_attentions=True
    )

    # ---------- AdamW curve ---------- #
    if args.train_log:
        plot_training_curves(
            args.train_log, os.path.join(args.output_dir, "adamw_training.png")
        )

    # ---------- Encode sample text ----- #
    inputs = tokenizer(args.sample_text, return_tensors="pt")
    with torch.no_grad():
        outputs_eval = model(**inputs, output_attentions=True)
        model.train()
        outputs_drop = model(**inputs, output_attentions=True)
    tok_labels = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # ---------- Heat‑maps side‑by‑side -- #
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_heatmap(
        aggregate_attention(
            outputs_eval.attentions, layer=args.layer, head=args.head, average=True
        ),
        tok_labels,
        ax[0],
        "Eval mode",
    )
    plot_heatmap(
        aggregate_attention(
            outputs_drop.attentions, layer=args.layer, head=args.head, average=True
        ),
        tok_labels,
        ax[1],
        "Train mode (dropout ON)",
    )
    fig.suptitle("Attention heat‑map comparison")
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "heatmap_compare.png"), dpi=300)
    plt.close(fig)

    # ---------- Top tokens bar ---------- #
    attn_eval = aggregate_attention(outputs_eval.attentions, average=True)
    top_attention_bar(
        attn_eval, tok_labels, os.path.join(args.output_dir, "top_attention_tokens.png")
    )

    # ---------- Attention roll‑out ------ #
    rollout_imp = attention_rollout(outputs_eval.attentions)
    tokens, scores = plot_rollout_importance(
        rollout_imp, tok_labels, os.path.join(args.output_dir, "rollout_importance.png")
    )
    # HTML highlighting
    html_file = html_highlight(
        tokens, scores, os.path.join(args.output_dir, "rollout_highlight.html")
    )
    print(f"Open {html_file} in a browser for colored text attribution.")


if __name__ == "__main__":
    main()
