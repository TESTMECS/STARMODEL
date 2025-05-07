"""
export_trainer_log.py
---------------------

Extracts the training & evaluation metrics stored in Hugging Face
`trainer_state.json` into a flat CSV (one row per log event).

Usage
-----

python export_trainer_log.py \
    --trainer_state star_bert_finetuned/trainer_state.json \
    --output train_history.csv \
    --keep loss eval_loss eval_f1

Arguments
---------
--trainer_state : path to trainer_state.json  (default: ./trainer_state.json)
--output        : where to write the CSV       (default: train_history.csv)
--keep          : space‑separated list of metric names to keep in the CSV.
                  If omitted, the script keeps **all** numeric fields.

The CSV will always include `step` as the first column.
"""

import argparse, json, csv, pathlib, sys


def parse_args():
    p = argparse.ArgumentParser(description="Convert trainer_state.json → CSV")
    p.add_argument(
        "--trainer_state",
        default="trainer_state.json",
        help="Path to trainer_state.json",
    )
    p.add_argument("--output", default="train_history.csv", help="Output CSV path")
    p.add_argument(
        "--keep",
        nargs="*",
        default=None,
        help="Metric names to keep (space‑separated). "
        "If omitted, all numeric fields are kept.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    state_path = pathlib.Path(args.trainer_state)
    if not state_path.is_file():
        sys.exit(f"❌  File not found: {state_path}")

    state = json.loads(state_path.read_text())
    log_history = state.get("log_history", [])

    # Collect field names dynamically
    fieldnames = {"step"}
    for rec in log_history:
        for k, v in rec.items():
            if k != "epoch" and isinstance(v, (int, float)):
                fieldnames.add(k)
    if args.keep:
        fieldnames = {"step"} | set(args.keep)
    fieldnames = sorted(fieldnames, key=lambda x: (x != "step", x))  # step first

    # Write CSV
    out_path = pathlib.Path(args.output)
    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for rec in log_history:
            if "step" not in rec:  # some eval logs lack 'step'
                continue
            row = {k: rec.get(k, "") for k in fieldnames}
            wr.writerow(row)

    print(f"✅  Wrote {out_path} with {sum(1 for _ in out_path.open())-1} rows.")


if __name__ == "__main__":
    main()
