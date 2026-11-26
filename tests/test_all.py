"""python tests/test_all.py   Checks the fact reader, metrics, study methods and app screens."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["ART_STUDY_DIR"] = tempfile.mkdtemp()  # never touch real study files

import numpy as np
import pandas as pd

import study_methods as sm
from facts import audit, extract_facts as ex
from pipeline import build_input, evaluate_system, field_table, load_records, template_summary, textrank_summary

multi = ("[2025-01-01] Continues TDF/3TC/EFV. CD4 150. Adherence poor. [2026-01-01] Regimen changed from "
         "TDF/3TC/EFV to TDF/3TC/DTG. CD4 600. Adherence good.")
f = ex(multi)
assert (f["regimen"], f["regimen_changed_from"], f["cd4"], f["adherence"]) == (
    "TDF/3TC/DTG", "TDF/3TC/EFV", "600", "good"), f
for text in ["No known drug allergies.", "Nil allergies.", "Allergies: none.", "NKDA."]:
    assert ex(text)["allergy"] == "none", text
assert ex("Viral load 12,000 copies/mL.")["viral_load"] == "12000"
assert ex("Viral load <40.")["viral_load"] == "<40"
assert ex("Viral load NOT suppressed.")["suppression"] == "not suppressed"
assert ex("Comorbidity: type 2 diabetes.")["comorbidity"] == "type 2 diabetes"
assert ex("[2025-01-01] Viral load 40. [2026-01-01] Viral load not suppressed.")["viral_load"] is None
results = {r["field"]: r["result"] for r in audit("Allergies: cotrimoxazole rash.", "Allergies: penicillin allergy.")}
assert results["allergy"] == "wrong"
assert {r["result"] for r in audit("CD4 500.", "Seen today.")} == {"unsupported"}

recs = load_records()
assert len(recs) == 500
keys = {"regimen": "current_regimen", "cd4": "cd4_cells_mm3", "adherence": "adherence", "tb_screen": "tb_screen",
        "allergy": "allergy", "comorbidity": "comorbidity", "next_appointment": "next_appointment"}
wrong = []
for r in recs:
    for text in (r["source_text"], template_summary(r)):
        got = ex(text)
        wrong += [(r["record_id"], k, got[k], r[c]) for k, c in keys.items() if got[k] != r[c]]
        vl = "not detected" if r["viral_load_copies_ml"] == "0" else r["viral_load_copies_ml"]
        if got["viral_load"] != vl:
            wrong.append((r["record_id"], "viral_load", got["viral_load"], vl))
    assert not [a for a in audit(template_summary(r), r["source_text"]) if a["result"] != "correct"
                and (a["field"], a["result"]) != ("viral_load_date", "missing")], r["record_id"]
assert not wrong, wrong[:5]
print("PASS facts: 500 of 500 synthetic records and their template summaries read correctly")

test = [r for r in recs if r["split"] == "test"]
for name, fn in (("template", template_summary), ("textrank", textrank_summary)):
    per, means = evaluate_system(test, [fn(r) for r in test], resamples=1000)
    print(name, {k: round(v[0], 3) for k, v in means.items()})
    print({t["field"]: round(t["accuracy"], 2) for t in field_table(per)})
text, cut = build_input(recs[0])
assert text.startswith("AGE") and cut == 0 and text.index("[2022") < text.index("[2021")

ones = pd.DataFrame([{k: 4 for k in sm.FEATURES} for _ in range(6)])
ones["mental"] = [4, 4, 4, 4, 1, 1]
freq = {k: 0.5 for k in sm.FEATURES}
freq["pregnancy"] = 0.0
d = sm.feature_decisions(ones, freq).set_index("key")
assert d.loc["mental", "Decision"] == "Not retained" and d.loc["pregnancy", "Decision"].startswith("Retained")
assert sm.feature_decisions(ones.head(5), freq)["Decision"].str.startswith("Pending").all()
plan = sm.allocate(15)
counts = pd.Series(plan.values()).value_counts()
assert counts.max() - counts.min() <= 1

rng = np.random.default_rng(1)  # invented numbers, only to check the statistics code runs
rows, rec_effect = [], {f"{s}{i}": rng.normal(0, .3) for s in "AB" for i in range(8)}
for c, seq in sm.allocate(15).items():
    ce = rng.normal(0, .2)
    for period, (cond, s) in enumerate(sm.SEQUENCES[seq], 1):
        for i in range(8):
            rows.append({"clinician": c, "record": f"{s}{i}", "condition": cond, "period": period,
                         "seconds": float(np.exp(4 + ce - .5 * (cond == "Assisted") + rec_effect[f"{s}{i}"] + rng.normal(0, .15))),
                         "score": float(np.clip(rng.normal(6, .7) + ce * 3 + rec_effect[f"{s}{i}"] * 2, 0, 8))})
res = sm.crossover_results(pd.DataFrame(rows))
assert 0.5 < res["time"]["ratio"] < 0.72 and res["comprehension"]["non_inferior"], res
assert abs(0.444 - evaluate_system(test, [template_summary(r) for r in test], resamples=100)[1]["rougeL"][0]) < .001
assert (sm.acceptance()["Met"] == "Pending").all()
print("PASS study methods")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
assert not at.exception, at.exception
btn = lambda label: next(b for b in at.button if b.label.startswith(label))
btn("Save ratings").click().run()
assert any("clinician code" in e.value for e in at.error)
next(t for t in at.text_input if t.label == "Clinician code").set_value("C01")
for radio in at.radio:
    if str(radio.key).startswith("f_"):
        radio.set_value(4)
btn("Save ratings").click().run()
assert any(s.value == "Saved." for s in at.success), [e.value for e in at.error]
at.text_input(key="gold_code").set_value("G1")
next(t for t in at.text_area if t.label == "Your summary").set_value("On TDF/3TC/DTG.")
btn("Save gold").click().run()
at.text_input(key="rater").set_value("R1").run()
for radio in at.radio:
    if radio.label in ("Completeness", "Correctness", "Conciseness"):
        radio.set_value(4)
    if radio.label == "Harm":
        radio.set_value(0)
btn("Save Summary A").click().run()
assert not at.exception, at.exception
btn("Open next record").click().run()
btn("Ready to see").click().run()
for t in at.text_input:
    if t.label[:2] in {f"{i}." for i in range(1, 9)}:
        t.set_value("not recorded")
btn("Save answers").click().run()
assert not at.exception, at.exception
btn("Save score").click().run()
btn("Run automatic").click().run()
assert not at.exception, at.exception
at.file_uploader[0].set_value(("t.csv", (ROOT / "test_data" / "art_hard_notes.csv").read_bytes(), "text/csv")).run()
assert not at.exception, at.exception
assert any(h.value.startswith("Accuracy against") for h in at.subheader)
study = Path(os.environ["ART_STUDY_DIR"])
assert sorted(p.name for p in study.glob("*.csv")) == [
    "feature_ratings.csv", "gold_summaries.csv", "ratings.csv", "reviews.csv"], list(study.iterdir())
assert pd.read_csv(study / "reviews.csv")["score"].notna().all()
print("PASS app: every screen saved data in a temporary folder\nALL TESTS PASSED")
