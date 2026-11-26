"""Fine tune BART-base or BioBART on the patient-level split (Sections 3.11.2 and 3.11.3).

  python train_bart.py --quick                      short check that everything runs
  python train_bart.py --seed 1                     one full run (repeat with seeds 2 and 3)
  python train_bart.py --fraction 0.25              learning curve (0.25, 0.5, 0.75, 1.0)
  python train_bart.py --search 20                  random search, scored on validation ROUGE-L
  python train_bart.py --gold study/gold_summaries.csv     silver plus gold arm
  python train_bart.py --zero-shot                  BART-large-CNN baseline, no training

Targets: gold where supplied, otherwise silver (the template). The synthetic CSV's
reference_summary is only used for scoring, never as a training target.
Test predictions are written ONLY with --final. Open the test set once, after all tuning."""
import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

import pandas as pd

from pipeline import ROOT, build_input, load_records, rouge_multi, template_summary

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="facebook/bart-base", help="or GanjinZero/biobart-base")
ap.add_argument("--data", default=str(ROOT / "data" / "art_synthetic_records.csv"))
ap.add_argument("--gold", help="CSV with record_id, split, summary")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--search", type=int, default=0)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--final", action="store_true")
ap.add_argument("--zero-shot", action="store_true")
a = ap.parse_args()

import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

recs = load_records(a.data)
split = {s: [r for r in recs if r["split"] == s] for s in ("train", "validation", "test")}
gold = {}
if a.gold:
    g = pd.read_csv(a.gold)
    gold = {k: list(v["summary"]) for k, v in g.groupby("record_id")}
if a.quick:
    split = {k: v[:16] for k, v in split.items()}


def references(r):
    return gold.get(r["record_id"], [r["reference_summary"]])


def target(r):
    return gold[r["record_id"]][0] if r["record_id"] in gold else template_summary(r)


def generate(model, tok, rows, beams, length_penalty=1.0, ngram=0):
    model.eval()
    out = []
    for i in range(0, len(rows), 8):
        enc = tok([build_input(r)[0] for r in rows[i:i + 8]], truncation=True, max_length=1024,
                  padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**enc, num_beams=beams, max_new_tokens=256, length_penalty=length_penalty,
                                 no_repeat_ngram_size=ngram)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def score(rows, summaries):
    return st.mean(rouge_multi(s, references(r))["rougeL"] for r, s in zip(rows, summaries))


def write_predictions(name, rows, summaries):
    (ROOT / "predictions").mkdir(exist_ok=True)
    pd.DataFrame({"record_id": [r["record_id"] for r in rows], "summary": summaries}).to_csv(
        ROOT / "predictions" / f"{name}.csv", index=False)


if a.zero_shot:
    name = "facebook/bart-large-cnn"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda" if torch.cuda.is_available() else "cpu")
    rows = split["test" if a.final else "validation"]
    out = generate(model, tok, rows, 4)
    print("ROUGE-L", round(score(rows, out), 3))
    if a.final:
        write_predictions("Zero-shot BART", rows, out)
    raise SystemExit


def train_once(cfg, out_dir):
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.checkpoint)
    rows = split["train"][:]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: max(1, int(len(rows) * a.fraction))]

    def encode(r):
        x = tok(build_input(r)[0], truncation=True, max_length=1024)
        x["labels"] = tok(text_target=target(r), truncation=True, max_length=256)["input_ids"]
        return dict(x)
    train, valid = [encode(r) for r in rows], [encode(r) for r in split["validation"]]
    accum = max(1, cfg["batch"] // 4)
    steps = math.ceil(len(train) / (4 * accum)) * cfg["epochs"]
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=cfg["epochs"], per_device_train_batch_size=4,
        per_device_eval_batch_size=4, gradient_accumulation_steps=accum, learning_rate=cfg["lr"],
        weight_decay=cfg["decay"], label_smoothing_factor=cfg["smooth"], lr_scheduler_type="linear",
        warmup_steps=int(0.1 * steps), eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, seed=a.seed, fp16=torch.cuda.is_available(), report_to=[])
    Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                   data_collator=DataCollatorForSeq2Seq(tok, model=model),
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    val = generate(model, tok, split["validation"], cfg["beams"], cfg["length"], cfg["ngram"])
    return model, tok, score(split["validation"], val)


base = {"lr": a.lr, "batch": 8, "epochs": 1 if a.quick else a.epochs, "decay": 0.01, "smooth": 0.0,
        "beams": 4, "length": 1.0, "ngram": 0}
tag = f"{a.checkpoint.split('/')[-1]}-{'gold' if gold else 'silver'}-f{a.fraction}-s{a.seed}"
out_dir = str(ROOT / "models" / tag)

if a.search:
    rng, log = random.Random(a.seed), []
    for t in range(a.search):
        cfg = {"lr": 10 ** rng.uniform(-5, -4), "batch": rng.choice([8, 16, 32]),
               "epochs": rng.choice([3, 5, 8, 10]), "decay": rng.choice([0.0, 0.01]),
               "smooth": rng.choice([0.0, 0.1]), "beams": rng.choice([2, 4, 6]),
               "length": rng.choice([1.0, 1.5, 2.0]), "ngram": rng.choice([0, 4])}
        _, _, val = train_once(cfg, f"{out_dir}-trial{t:02d}")
        log.append({**cfg, "trial": t, "val_rougeL": val})
        pd.DataFrame(log).to_csv(ROOT / "search_log.csv", index=False)
    print(max(log, key=lambda r: r["val_rougeL"]))
else:
    model, tok, val = train_once(base, out_dir)
    Path(out_dir, "run.json").write_text(json.dumps({**vars(a), **base, "val_rougeL": val}, indent=1))
    print(f"Validation ROUGE-L {val:.3f}. Model saved in {out_dir}")
    if a.final:
        test = split["test"]
        write_predictions(f"Fine tuned BART {tag}", test,
                          generate(model, tok, test, base["beams"], base["length"], base["ngram"]))
        print("Test predictions written to the predictions folder.")
