"""Final test-set comparison (Sections 3.12.1, 3.12.2 and criteria A and B).

  python evaluate_final.py --model "predictions/Fine tuned BART ....csv" \
        --gold study/gold_summaries.csv --ratings study/ratings.csv

Needs: pip install bert-score   (for BERTScore, criterion A)"""
import argparse
import statistics as st

import pandas as pd

import study_methods as sm
from pipeline import SYSTEMS, bootstrap_ci, evaluate_system, field_table, load_records, rouge_multi

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--ratings")
a = ap.parse_args()

gold = pd.read_csv(a.gold)
gold = gold[gold["split"] == "test"]
refs = {k: list(v["summary"]) for k, v in gold.groupby("record_id")}
records = [r for r in load_records() if r["record_id"] in refs]
if not records:
    raise SystemExit("No gold test summaries yet. Nothing to report.")
preds = dict(pd.read_csv(a.model)[["record_id", "summary"]].values)
missing = [r["record_id"] for r in records if r["record_id"] not in preds]
if missing:
    raise SystemExit(f"The model file lacks predictions for {missing[:5]}")

outputs = {n: [f(r) for r in records] for n, f in SYSTEMS.items()}
outputs["Model"] = [str(preds[r["record_id"]]) for r in records]
reflist = [refs[r["record_id"]] for r in records]
results = {n: evaluate_system(records, s, reflist) for n, s in outputs.items()}

try:
    from bert_score import score as bert
    for n, summaries in outputs.items():
        each = [bert([s] * len(r), r, lang="en")[2].mean().item() for s, r in zip(summaries, reflist)]
        for p, v in zip(results[n][0], each):
            p["bertscore"] = v
except ImportError:
    print("bert-score is not installed, so criterion A stays Pending.")

for n, (per, means) in results.items():
    print(f"\n== {n} on {len(records)} gold test records")
    for k, (m, lo, hi) in means.items():
        print(f"{k}: {m:.3f} ({lo:.3f} to {hi:.3f})")
    if "bertscore" in per[0]:
        print("bertscore:", round(st.mean(p["bertscore"] for p in per), 3))
    print(pd.DataFrame(field_table(per)).round(3).to_string(index=False))

two = [v for v in reflist if len(v) >= 2]
if two:
    ceiling = [(rouge_multi(x, [y])["rougeL"] + rouge_multi(y, [x])["rougeL"]) / 2 for x, y, *_ in two]
    print("\nHuman ceiling ROUGE-L:", round(st.mean(ceiling), 3))

crit_a = crit_b = None
if "bertscore" in results["Model"][0]:
    lows = []
    for metric in ("rougeL", "bertscore"):
        best = max(SYSTEMS, key=lambda n: st.mean(p[metric] for p in results[n][0]))
        diff = [m[metric] - b[metric] for m, b in zip(results["Model"][0], results[best][0])]
        lows.append(bootstrap_ci(diff)[0])
    crit_a = tuple(lows)
if a.ratings:
    r = pd.read_csv(a.ratings)
    r = r[r["system"].str.contains("bart", case=False) & r["record"].isin(refs)]
    acc = {t["field"]: t["accuracy"] for t in field_table(results["Model"][0])}
    covered = set(r["record"]) >= set(refs)
    if covered and acc.get("regimen") is not None and acc.get("viral_load") is not None:
        serious = int(((r["harm"] == 3) & (r["regimen_or_vl_harm"] == 1)).sum())
        crit_b = (acc["regimen"], acc["viral_load"], serious)
    else:
        print("Criterion B stays Pending: not every test record has a harm rating.")
print("\n" + sm.acceptance(a=crit_a, b=crit_b).iloc[:2].to_string(index=False))
print("The audit is rule based. A clinician must check what the script cannot resolve.")
