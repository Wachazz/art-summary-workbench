"""Make a MADE-UP test file with an answer key. Wording is deliberately more varied than the training generator.
No real patient is used. Run: python make_test_data.py"""
import csv, random
from datetime import date, timedelta
from pipeline import ROOT

rng = random.Random(77)
REG = ["TDF/3TC/DTG", "TDF/3TC/EFV", "ABC/3TC/DTG", "AZT/3TC/DTG"]
FILL = ["Came with treatment supporter.", "Queue long today.", "BP 120/80.", "Weight 61 kg.", "Pill count done."]
rows = []
for i in range(40):
    old, new = rng.sample(REG, 2)
    d1 = date(2025, 1, 6) + timedelta(days=rng.randint(0, 60))
    d2 = d1 + timedelta(days=rng.randint(80, 100))
    d3 = d2 + timedelta(days=rng.randint(80, 100))
    nxt = d3 + timedelta(days=90)
    switched = rng.random() < .5
    cur = new if switched else old
    vl = rng.choice(["<50", "40", "380", "1240", "12000", "56000"])
    cd4, adh = str(rng.randint(120, 800)), rng.choice(["good", "fair", "poor"])
    tb = rng.choice(["negative", "positive"])
    al = rng.choice(["none", "none", "penicillin allergy", "cotrimoxazole rash"])
    reg_line = (rng.choice([f"Regimen changed from {old} to {new} due to side effects.",
                            f"Switched from {old} to {new}.",
                            f"Stopped {old}. Started {new} today."]) if switched else f"Continues {old}.")
    vl_txt = rng.choice([f"Viral load {int(vl.lstrip('<')):,} copies/mL" if vl != "<50" else "Viral load <50 copies/mL",
                         f"VL: {vl}", f"VL result {vl}"])
    al_txt = (rng.choice(["No known drug allergies.", "Nil allergies.", "Allergies: none.", "NKDA."]) if al == "none"
              else rng.choice([f"Allergies: {al}.", f"Allergic to {al.split()[0]}."]))
    latest = " ".join([rng.choice(FILL), vl_txt + ".",
                       rng.choice([f"CD4 {cd4}.", f"CD4 count: {cd4}.", f"CD4={cd4}."]),
                       rng.choice([f"Adherence {adh}.", f"Adherence was {adh}.", f"{adh.title()} adherence."]),
                       rng.choice([f"TB screen {tb}.", f"TB screening was {tb}.", f"TB screen: {tb}."]), al_txt,
                       rng.choice([f"Next appointment {nxt}.", f"Review on {nxt}.", f"Next visit: {nxt}."])])
    text = (f"[{d1}] {rng.choice(FILL)} Continues {old}. CD4 {rng.randint(100, 900)}. Adherence {rng.choice(['good','poor'])}. "
            f"[{d2}] {reg_line} {rng.choice(FILL)} [{d3}] {latest}")
    rows.append({"record_id": f"TEST-{i+1:03d}", "source_text": text, "expected_regimen": cur,
                 "expected_viral_load": vl, "expected_cd4": cd4, "expected_adherence": adh, "expected_tb_screen": tb,
                 "expected_allergy": al, "expected_next_appointment": str(nxt)})
(ROOT / "test_data").mkdir(exist_ok=True)
with open(ROOT / "test_data" / "art_test_dataset.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("40 made-up records written")
