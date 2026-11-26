"""Data loading, summarisers, input construction and automatic metrics."""
import csv
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from facts import audit, extract_facts, visits_oldest_first

ROOT = Path(__file__).parent
SENT = re.compile(r"(?<=[.!?])\s+(?=[\[A-Z])")
PATTERNS = {
    "regimen": r"\b(TDF|AZT|ABC|3TC|DTG|EFV|TLD)\b|regimen",
    "viral_load": r"viral load|\bVL\b",
    "cd4": r"\bCD4\b",
    "adherence": r"adheren|missed|\bEAC\b",
    "tb": r"\bTB\b|\bTPT\b",
    "allergy": r"allerg",
    "comorbidity": r"comorbid|hypertension|diabetes|asthma",
    "pregnancy": r"pregnan|breastfeed|\bPMTCT\b",
    "mental_social": r"depress|alcohol|stigma|disclos|food|supporter",
    "pending": r"next appointment|review again|referr|pending|due",
}


def load_records(path=ROOT / "data" / "art_synthetic_records.csv"):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ summarisers
def template_summary(rec):
    """Template baseline written from the coded fields (Section 3.11.4)."""
    vl = "not detected" if rec["viral_load_copies_ml"] == "0" else f"{rec['viral_load_copies_ml']} copies/mL"
    # wording kept identical to the version behind the Chapter 4 scores
    out = [f"Patient is on {rec['current_regimen']}.", f"Viral load {vl}. CD4 {rec['cd4_cells_mm3']}.",
           f"Adherence {rec['adherence']}. TB screen {rec['tb_screen']}.",
           f"Allergies {rec['allergy']}. Comorbidity {rec['comorbidity']}."]
    if rec["regimen_changed_from"] not in ("", "none"):
        out.append(f"Regimen changed from {rec['regimen_changed_from']}.")
    out.append(f"Next appointment {rec['next_appointment']}.")
    return " ".join(out)


def sentences(text):
    return [s.strip() for s in SENT.split(text) if s.strip()]


def textrank_summary(rec, k=4):
    s = sentences(rec["source_text"])
    if len(s) <= k:
        return " ".join(s)
    graph = nx.from_numpy_array(cosine_similarity(TfidfVectorizer().fit_transform(s)))
    rank = nx.pagerank(graph)
    return " ".join(s[i] for i in sorted(sorted(rank, key=rank.get, reverse=True)[:k]))


SYSTEMS = {"Template baseline": template_summary, "TextRank": textrank_summary}


# ------------------------------------------------------------------ model input (3.9)
def build_input(rec, max_words=650, retained=None):
    """Coded block first, then notes newest first. When space runs out, sentences matching
    retained features are kept before other sentences. max_words stands in for the token
    limit. train_bart.py truncates at 1,024 real tokens as the final guard.
    Returns (text, number of feature sentences that had to be cut)."""
    prev = rec.get("regimen_changed_from", "none")
    coded = [f"AGE {rec.get('age', '')} | SEX {rec.get('sex', '')}",
             f"REGIMEN NOW {rec['current_regimen']} | PREVIOUS {prev}",
             f"VL LATEST {rec['viral_load_copies_ml']} copies/mL on {rec.get('last_visit_date', '')}",
             f"CD4 LATEST {rec['cd4_cells_mm3']} | ADHERENCE {rec['adherence']} | TB SCREEN {rec['tb_screen']}",
             f"ALLERGY {rec['allergy']} | COMORBIDITY {rec['comorbidity']} | NEXT {rec['next_appointment']}",
             "NOTES newest first"]
    pats = [p for key, p in PATTERNS.items() if retained is None or key in retained]
    items = []
    for vi, (date, body) in enumerate(reversed(visits_oldest_first(rec["source_text"]))):
        for si, s in enumerate(sentences(body)):
            hit = any(re.search(p, s, re.I) for p in pats)
            items.append((0 if hit else 1, vi, si, date, s))
    used, keep, cut = sum(len(c.split()) for c in coded), set(), 0
    for hit_rank, vi, si, date, s in sorted(items):
        n = len(s.split()) + 1
        if used + n <= max_words:
            used += n
            keep.add((vi, si))
        elif hit_rank == 0:
            cut += 1
    lines, last = [], None
    for _, vi, si, date, s in sorted(items, key=lambda t: (t[1], t[2])):
        if (vi, si) in keep:
            if vi != last:
                lines.append(f"[{date}] {s}")
                last = vi
            else:
                lines[-1] += " " + s
    return "\n".join(coded + lines), cut


# ------------------------------------------------------------------ metrics (3.12.1)
def _tok(t):
    return re.findall(r"[a-z0-9/\.\-]+", t.lower())


def _f1(overlap, a, b):
    p, r = overlap / max(a, 1), overlap / max(b, 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def rouge(pred, ref):
    """Built-in ROUGE with no stemming. Report the rouge-score package for the final study."""
    x, y, out = _tok(pred), _tok(ref), {}
    for n in (1, 2):
        a = Counter(tuple(x[i:i + n]) for i in range(len(x) - n + 1))
        b = Counter(tuple(y[i:i + n]) for i in range(len(y) - n + 1))
        out[f"rouge{n}"] = _f1(sum((a & b).values()), sum(a.values()), sum(b.values()))
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    out["rougeL"] = _f1(prev[-1], len(x), len(y))
    return out


def rouge_multi(pred, refs):
    """Score against each gold reference separately and average."""
    each = [rouge(pred, r) for r in refs]
    return {k: st.mean(e[k] for e in each) for k in each[0]}


def bootstrap_ci(values, resamples=10000, seed=2026):
    """95% interval of the mean, resampling patients (one value per patient)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def evaluate_system(records, summaries, refs=None, resamples=10000):
    """summaries: list aligned with records. refs: list of lists of gold references."""
    refs = refs or [[r["reference_summary"]] for r in records]
    per = []
    for rec, summ, rf in zip(records, summaries, refs):
        per.append({"record": rec["record_id"], **rouge_multi(summ, rf),
                    "audit": audit(summ, rec["source_text"]), "words": len(summ.split())})
    means = {}
    for k in ("rouge1", "rouge2", "rougeL"):
        vals = [p[k] for p in per]
        means[k] = (st.mean(vals), *bootstrap_ci(vals, resamples))
    return per, means


def field_table(per):
    """Field accuracy, omission rate and errors per 100 summaries."""
    rows = {}
    for p in per:
        for a in p["audit"]:
            rows.setdefault(a["field"], Counter())[a["result"]] += 1
    table = []
    for field, c in rows.items():
        in_source = c["correct"] + c["wrong"] + c["missing"]
        table.append({"field": field, "records": in_source,
                      "accuracy": c["correct"] / in_source if in_source else None,
                      "omission": c["missing"] / in_source if in_source else None,
                      "wrong_per_100": 100 * c["wrong"] / len(per),
                      "unsupported_per_100": 100 * c["unsupported"] / len(per)})
    return table
