# STAR Sentence Classifier 📄
```bash
python star_trainer.py \
  --csv_file data/cleaned_df.csv \
  --model_name google-bert/bert-base-uncased \
  --epochs 8 --lr 3e-5 --batch_size 32
```
A lightning‑quick way to **train, evaluate, and analyse** a classifier that tags interview sentences with their STAR stage—**Situation, Task, Action, Result**—using any Hugging Face transformer.

---

## 🚀 Quick Start

```bash
# 1. Create & activate a Python ≥3.10 env
uv sync && source .venv/bin/activate

# 2. Train 🏋️‍♀️
python star_trainer.py \
  --csv_file data/annotated.csv \
  --model_name google-bert/bert-base-uncased \
  --epochs 4 \
  --lr 3e-5 \
  --batch_size 32
```

Outputs land in `star_output/` (or your `--output_dir`):

```
loss_curve.png           # train vs eval loss per epoch
accuracy_curve.png       # eval accuracy per epoch
confusion_matrix.png     # 4×4 heat‑map
metrics.md               # copy‑paste table for your Results section
best_model/              # saved HF model & tokenizer
```

---

## 🛠️  CLI reference

| Flag           | Default             | Description                           |
| -------------- | ------------------- | ------------------------------------- |
| `--csv_file`   | **required**        | CSV with cols `sentence,label`        |
| `--model_name` | `bert-base-uncased` | HF model checkpoint                   |
| `--output_dir` | `star_output`       | Where checkpoints & plots are written |
| `--epochs`     | `5`                 | Training epochs                       |
| `--lr`         | `5e-5`              | AdamW learning‑rate                   |
| `--batch_size` | `16`                | Per‑device batch size                 |
| `--max_length` | `128`               | Token truncation length               |
| `--seed`       | `42`                | RNG seed for splits & PyTorch         |

### Recommended hyper‑parameter presets

| Model                     | LR     | Batch | Notes                                   |
| ------------------------- | ------ | ----- | --------------------------------------- |
| `bert-base-uncased`       | `5e-5` | 16    | Good baseline. <2 GB VRAM.              |
| `roberta-base`            | `3e-5` | 32    | Slightly better F1. Needs \~8 GB VRAM.  |
| `deberta-v3-small`        | `2e-5` | 32    | Strong performance, slower training.    |
| `distilbert-base-uncased` | `8e-5` | 32    | Lightweight, use for rapid prototyping. |

> **Tip 💡**: Reduce `--batch_size` if you hit CUDA OOM; compensate by increasing `--epochs`.

---

## 🧪 Adding extras

* **Early stopping**: wrap `transformers.EarlyStoppingCallback` and add to `Trainer`.
* **CRF head**: replace the classification layer with `torchcrf.CRF` to model label order.
* **Cross‑validation**: loop over folds in a shell or Python wrapper.
* **Data augmentation**: back‑translate sentences or use synonym replacement.
* **Scheduler tweaks**: swap in `get_cosine_schedule_with_warmup` or a linear decay.
* **Inference**: load `best_model/` and call `model(**tokenizer(text, return_tensors="pt"))`.

---

## 📁 Project layout

```
.
├── data/                  # Your CSV lives here
├── star_trainer.py        # Main training script (class‑based)
├── README.md              # You are here 👋
└── star_output/           # Auto‑generated artefacts
```

Made with ❤️
