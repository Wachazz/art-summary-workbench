"""ART Summary Workbench, version 2. Follows the three objectives of the dissertation.
Run:  streamlit run streamlit_app.py
Synthetic data only. Do not enter real patient records before the approvals in Section 3.14."""
import os
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import study_methods as sm
from facts import LABELS, audit, extract_facts
from score_file import score_file
from pipeline import (SYSTEMS, build_input, evaluate_system, field_table, load_records, rouge_multi)

ROOT = Path(__file__).parent
STUDY = Path(os.environ.get("ART_STUDY_DIR", ROOT / "study"))
STUDY.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ART Summary Workbench", layout="wide")


def read(name, cols):
    p = STUDY / name
    return pd.read_csv(p, keep_default_na=False, na_values=[""]) if p.exists() else pd.DataFrame(columns=cols)


def save(name, row):
    """Rewrite the whole file so new columns never misalign old rows."""
    p = STUDY / name
    df = pd.DataFrame([row])
    if p.exists():
        df = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=[""]), df], ignore_index=True)
    df.to_csv(p, index=False)


def download(df, name):
    st.download_button(f"Download {name}", df.to_csv(index=False), name, "text/csv", key=f"dl_{name}")


@st.cache_data
def records():
    return load_records()


@st.cache_resource
def load_model(path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    return AutoTokenizer.from_pretrained(path), AutoModelForSeq2SeqLM.from_pretrained(path)


recs = records()
by_id = {r["record_id"]: r for r in recs}
test = [r for r in recs if r["split"] == "test"]

st.sidebar.title("ART Summary Workbench")
st.sidebar.warning("Synthetic data only. Not for clinical use.")
systems = dict(SYSTEMS)
pred_files = sorted((ROOT / "predictions").glob("*.csv"))
for f in pred_files:  # model outputs written by train_bart.py
    preds = dict(pd.read_csv(f)[["record_id", "summary"]].values)
    systems[f.stem] = lambda r, p=preds: p.get(r["record_id"], "")
model_dir = st.sidebar.text_input("Trained BART folder (optional)", "")
if model_dir:
    if not Path(model_dir).is_dir():
        st.sidebar.error("That folder does not exist. Train the model first.")
    else:
        try:
            tok, model = load_model(model_dir)

            def bart(r):
                enc = tok(build_input(r)[0], truncation=True, max_length=1024, return_tensors="pt")
                ids = model.generate(**enc, num_beams=4, max_new_tokens=256)
                return tok.decode(ids[0], skip_special_tokens=True)
            systems["Fine tuned BART"] = bart
        except Exception as e:
            st.sidebar.error(f"Model could not load: {e}")
if not any("bart" in s.lower() for s in systems):
    st.sidebar.info("BART: not trained yet. No BART results are shown.")

tabs = st.tabs(["1. Essential features", "2. Summariser", "Gold summaries", "3a. Clinician ratings",
                "3b. Review time", "Results", "Test a file"])

# ---------------------------------------------------------------- objective 1
with tabs[0]:
    st.header("Research question 1: which ART features are essential?")
    st.write("Would this belong in a pre-consultation summary? 1 not relevant, 2 somewhat relevant, "
             "3 quite relevant, 4 highly relevant.")
    with st.form("features"):
        code = st.text_input("Clinician code", placeholder="C01")
        values = {}
        for k, (label, examples, _, safety) in sm.FEATURES.items():
            values[k] = st.radio(f"{label}{' (safety critical)' if safety else ''}", [1, 2, 3, 4],
                                 index=None, horizontal=True, help=examples, key=f"f_{k}")
        extra = st.text_input("A feature we missed (optional)")
        if st.form_submit_button("Save ratings"):
            old = read("feature_ratings.csv", ["clinician"])
            if not code.strip():
                st.error("Enter a clinician code.")
            elif code.strip() in set(old["clinician"].astype(str)):
                st.error("That clinician code has already rated.")
            elif any(v is None for v in values.values()):
                st.error("Rate every feature. Nothing was saved.")
            else:
                save("feature_ratings.csv", {"clinician": code.strip(), **values, "other": extra,
                                             "time": datetime.now().isoformat(timespec="seconds")})
                st.success("Saved.")
    ratings = read("feature_ratings.csv", ["clinician", *sm.FEATURES])
    table = sm.feature_decisions(ratings, sm.document_frequency([r["source_text"] for r in recs]))
    st.subheader(f"Ranked list ({len(ratings)} of {sm.PANEL_TARGET} planned raters)")
    st.dataframe(table.drop(columns="key"), hide_index=True, width="stretch")
    st.caption("Kept when I-CVI is 0.78 or higher and the feature is in 5% or more of records. Safety critical "
               "items are always kept. 'In records' is measured on synthetic notes.")
    if len(ratings):
        download(ratings, "feature_ratings.csv")

# ---------------------------------------------------------------- objective 2
with tabs[1]:
    st.header("Research question 2: summarising the record")
    mode = st.radio("Source", ["Synthetic record", "Paste a synthetic note"], horizontal=True)
    if mode == "Synthetic record":
        rec = by_id[st.selectbox("Record", [r["record_id"] for r in test], help="The 50 test records")]
        source = rec["source_text"]
    else:
        rec = None
        source = st.text_area("Note, with a [YYYY-MM-DD] date before each visit",
                              "[2026-04-02] Continues TDF/3TC/EFV. CD4 150. Adherence poor. "
                              "[2026-10-12] Regimen changed from TDF/3TC/EFV to TDF/3TC/DTG. Viral load 12,000 "
                              "copies/mL. CD4 412. Adherence good. TB screen negative. No known drug allergies. "
                              "Next appointment 2027-01-11.", height=150)
    with st.expander("Source record", expanded=True):
        st.write(source)
    st.subheader("Latest facts found in the notes")
    found = extract_facts(source)
    st.dataframe(pd.DataFrame([{"Field": LABELS[k], "Value": v or "not found"} for k, v in found.items()]),
                 hide_index=True, width="stretch")
    if rec:
        text, cut = build_input(rec)
        with st.expander("Model input (coded block, then notes newest first)"):
            st.code(text, language=None)
            st.caption(f"Feature sentences cut for length: {cut}")
        cols = st.columns(len(systems))
        for col, (name, fn) in zip(cols, systems.items()):
            with col:
                summary = fn(rec)
                st.subheader(name)
                st.write(summary or "No output for this record.")
                rows = audit(summary, source)
                good = sum(r["result"] == "correct" for r in rows)
                st.caption(f"Audit: {good} of {len(rows)} fields correct")
                st.dataframe(pd.DataFrame(rows).assign(field=lambda d: d["field"].map(LABELS)),
                             hide_index=True, width="stretch")
    else:
        st.info("Pasted notes show extracted facts only. The rules can miss wording, so check the source.")

# ---------------------------------------------------------------- gold
with tabs[2]:
    st.header("Gold summaries written by clinicians")
    st.write("Write what is needed in the two minutes before the patient arrives. No more than 120 words, "
             "no identifiers. For validation and test records, no model output is shown.")
    c1, c2, c3 = st.columns(3)
    author = c1.text_input("Clinician code", key="gold_code")
    split = c2.selectbox("Set", ["test", "validation", "train"])
    rid = c3.selectbox("Record", [r["record_id"] for r in recs if r["split"] == split], key="gold_rec")
    st.text_area("Source record", by_id[rid]["source_text"], height=180, disabled=True, key=f"gs_{rid}")
    draft = SYSTEMS["Template baseline"](by_id[rid]) if split == "train" else ""
    gold_text = st.text_area("Your summary", draft, height=130, key=f"gold_{rid}")
    if st.button("Save gold summary"):
        n = len(gold_text.split())
        old = read("gold_summaries.csv", ["author", "record_id"])
        if not author.strip() or not n:
            st.error("Enter your code and a summary.")
        elif n > 120:
            st.error(f"{n} words. The limit is 120.")
        elif ((old["author"].astype(str) == author.strip()) & (old["record_id"] == rid)).any():
            st.error("You have already saved a summary for this record.")
        else:
            save("gold_summaries.csv", {"author": author.strip(), "record_id": rid, "split": split,
                                        "summary": gold_text.strip()})
            st.success("Saved.")
    gold = read("gold_summaries.csv", ["author", "record_id", "split", "summary"])
    st.caption(f"Saved: {len(gold)}. Planned: 120 test (two per record for 60), 60 validation, 100 training.")
    if len(gold):
        download(gold, "gold_summaries.csv")

# ---------------------------------------------------------------- ratings
with tabs[3]:
    st.header("Research question 3: clinician ratings (blind)")
    rater = st.text_input("Rater code", key="rater", placeholder="R1")
    rid = st.selectbox("Record", [r["record_id"] for r in test], key="rate_rec")
    options = {n: f(by_id[rid]) for n, f in systems.items()}
    gold = read("gold_summaries.csv", ["author", "record_id", "split", "summary"])
    g = gold[gold["record_id"] == rid]
    if len(g):
        options["Gold summary"] = g.iloc[0]["summary"]
    names = list(options)
    random.Random(f"{rater}|{rid}").shuffle(names)
    st.text_area("Source record", by_id[rid]["source_text"], height=160, disabled=True, key=f"rs_{rid}")
    done = read("ratings.csv", ["rater", "record", "system"])
    for i, name in enumerate(names):
        label = f"Summary {chr(65 + i)}"
        if ((done["rater"].astype(str) == rater.strip()) & (done["record"] == rid) & (done["system"] == name)).any():
            st.success(f"{label}: rated.")
            continue
        with st.form(f"rate_{rid}_{i}_{rater}"):
            st.markdown(f"**{label}**")
            st.write(options[name] or "(empty summary)")
            cols = st.columns(4)
            scores = [cols[j].radio(q, [1, 2, 3, 4, 5], index=None, horizontal=True)
                      for j, q in enumerate(["Completeness", "Correctness", "Conciseness"])]
            harm = cols[3].radio("Harm", [0, 1, 2, 3], index=None, horizontal=True, help="0 none, 3 serious")
            cause = st.checkbox("Harm is from a wrong or invented regimen or viral load")
            flags = st.multiselect("Sensitive content", ["Dropped disclosure or stigma concern",
                                                         "Judgemental phrasing", "Identifying detail"])
            if st.form_submit_button(f"Save {label}"):
                others = done[(done["record"] == rid) & (done["system"] == name)]["rater"].nunique()
                if not rater.strip():
                    st.error("Enter a rater code.")
                elif None in scores or harm is None:
                    st.error("Answer all four scales.")
                elif others >= 2:
                    st.error("This item already has two raters. Choose another record.")
                else:
                    save("ratings.csv", {"rater": rater.strip(), "record": rid, "system": name,
                                         "completeness": scores[0], "correctness": scores[1],
                                         "conciseness": scores[2], "harm": harm,
                                         "regimen_or_vl_harm": int(cause), "flags": "; ".join(flags)})
                    st.rerun()

# ---------------------------------------------------------------- crossover
with tabs[4]:
    st.header("Research question 3: review time and comprehension")
    plan = sm.allocate(15)
    c1, c2, c3 = st.columns(3)
    code = c1.selectbox("Clinician code", list(plan))
    period = c2.selectbox("Period", [1, 2])
    condition, setname = sm.SEQUENCES[plan[code]][period - 1]
    chosen = random.Random(2026).sample([r["record_id"] for r in test], 16)
    pool = chosen[:8] if setname == "A" else chosen[8:]
    st.info(f"Sequence {plan[code]}. Period {period}: {condition} review of set {setname}. "
            "Leave at least one working day between periods.")
    sysname = c3.selectbox("Summary system", list(systems), index=len(systems) - 1 if len(systems) > 2 else 0,
                           disabled=condition == "Manual")
    st.caption("Sets are random, not yet matched on care stage and length (Section 3.12.4).")
    reviews = read("reviews.csv", ["review_id", "clinician", "record"])
    finished = set(reviews[reviews["clinician"] == code]["record"])
    left = [r for r in pool if r not in finished]
    state = st.session_state.get("review")
    if state is None:
        if not left:
            st.success("All eight records in this period are done.")
        else:
            st.write(f"{8 - len(left)} of 8 done.")
            if st.button("Open next record", type="primary"):
                summary = systems[sysname](by_id[left[0]]) if condition == "Assisted" else ""
                st.session_state["review"] = {"clinician": code, "period": period, "condition": condition,
                                              "set": setname, "record": left[0], "summary": summary,
                                              "system": sysname if condition == "Assisted" else "",
                                              "start": time.time(), "seconds": None}
                st.rerun()
    elif state["seconds"] is None:
        st.markdown(f"**{state['clinician']} | {state['condition']} | {state['record']}**")
        if state["summary"]:
            st.success(state["summary"])
        st.write(by_id[state["record"]]["source_text"])
        if st.button("Ready to see the patient", type="primary"):
            state["seconds"] = round(time.time() - state["start"], 1)
            st.rerun()
        if st.button("Cancel this review"):
            del st.session_state["review"]
            st.rerun()
    else:
        st.write("Answer from memory. The record is now hidden. Write 'not recorded' if it was not there.")
        with st.form("questions"):
            answers = [st.text_input(f"{i + 1}. {q}") for i, q in enumerate(sm.QUESTIONS)]
            if st.form_submit_button("Save answers"):
                if not all(a.strip() for a in answers):
                    st.error("Answer all eight questions.")
                else:
                    row = {k: state[k] for k in ("clinician", "period", "condition", "set", "record",
                                                 "system", "seconds")}
                    row = {"review_id": f"{state['clinician']}-{state['record']}", **row,
                           **{f"q{i + 1}": a for i, a in enumerate(answers)}, "score": None}
                    save("reviews.csv", row)
                    del st.session_state["review"]
                    st.rerun()
    with st.expander("Scorer only: mark the answers"):
        reviews = read("reviews.csv", ["review_id"])
        waiting = reviews[reviews["score"].isna()] if "score" in reviews else reviews
        if not len(waiting):
            st.write("Nothing to score.")
        else:
            rid = st.selectbox("Review", list(waiting["review_id"]))
            row = reviews[reviews["review_id"] == rid].iloc[0]
            key = by_id[row["record"]]
            st.write("Coded answer key:", {k: key[k] for k in (
                "current_regimen", "regimen_changed_from", "last_visit_date", "viral_load_copies_ml",
                "viral_load_status", "tb_screen", "adherence", "next_appointment")})
            st.text(key["source_text"])
            st.caption("Questions 6 to 8 are marked against the reading agreed by the panel.")
            with st.form(f"score_{rid}"):
                marks = [st.checkbox(f"{i + 1}. {q}  Answer: {row[f'q{i + 1}']}") for i, q in enumerate(sm.QUESTIONS)]
                if st.form_submit_button("Save score (ticked answers are correct)"):
                    reviews.loc[reviews["review_id"] == rid, "score"] = sum(marks)
                    reviews.to_csv(STUDY / "reviews.csv", index=False)
                    st.rerun()

# ---------------------------------------------------------------- results
with tabs[5]:
    st.header("Results")
    st.warning("Scores below use synthetic records and rule-written references. They test the pipeline and "
               "say nothing about clinical performance.")
    if st.button("Run automatic metrics and factual audit on the 50 test records"):
        for name, fn in systems.items():
            summaries = [fn(r) for r in test]
            per, means = evaluate_system(test, summaries)
            st.subheader(name)
            st.dataframe(pd.DataFrame([{"Metric": k, "Mean": round(v[0], 3), "95% low": round(v[1], 3),
                                        "95% high": round(v[2], 3)} for k, v in means.items()]), hide_index=True)
            st.dataframe(pd.DataFrame(field_table(per)).assign(field=lambda d: d["field"].map(LABELS)).round(3),
                         hide_index=True)
        st.caption("The template baseline writes from the coded fields, so its high accuracy holds by construction.")
    st.subheader("Clinician ratings")
    r = read("ratings.csv", ["rater", "record", "system"])
    if len(r):
        st.dataframe(r.groupby("system")[["completeness", "correctness", "conciseness", "harm"]]
                     .agg(["mean", "std", "count"]).round(2))
        st.write("Weighted kappa:", {c: sm.weighted_kappa(r, c) or "Pending"
                                     for c in ("completeness", "correctness", "conciseness", "harm")})
        download(r, "ratings.csv")
    else:
        st.write("Pending. No ratings yet.")
    st.subheader("Review time and comprehension")
    v = read("reviews.csv", ["review_id", "clinician", "record", "condition", "period", "seconds", "score"])
    res = {"note": "", "time": None, "comprehension": None}
    if len(v):
        try:
            res = sm.crossover_results(v)
        except Exception as e:
            res["note"] = f"The model could not be fitted yet: {e}"
        st.dataframe(v.groupby("condition")[["seconds", "score"]].mean().round(2))
        st.write(res["note"])
        if res["time"]:
            st.write("Review time, assisted to manual ratio:", res["time"])
            st.write("Wilcoxon and Cohen's dz:", sm.paired_checks(v))
        if res["comprehension"]:
            st.write("Comprehension, assisted minus manual:", res["comprehension"])
        download(v, "reviews.csv")
    else:
        st.write("Pending. No reviews yet.")
    st.subheader("Acceptance criteria")
    c = d = None
    model_name = next((n for n in systems if "bart" in n.lower()), None)
    mr = r[r["system"] == model_name] if len(r) else r
    if model_name and len(mr):
        c = (mr["correctness"].mean(), mr["completeness"].mean())
    if (model_name and res["time"] and res["comprehension"] and v["clinician"].nunique() >= 12
            and set(v.loc[v["condition"] == "Assisted", "system"]) == {model_name}):
        d = (res["time"]["high"], res["time"]["p"], res["comprehension"]["low"])
    st.dataframe(sm.acceptance(c=c, d=d), hide_index=True, width="stretch")
    st.caption("A and B are decided from the final test-set evaluation with gold summaries and the trained model "
               "(run evaluate_final.py). Nothing is marked as met without that evidence.")

# ---------------------------------------------------------------- test a file
with tabs[6]:
    st.header("Test the fact reader on a file")
    st.error("Made-up data only until the approvals in Section 3.14. Never upload real patient records to a website.")
    st.write("The CSV needs record_id and source_text columns, with a [YYYY-MM-DD] date before each visit. "
             "Add expected_regimen, expected_viral_load, expected_cd4 and similar columns to get an accuracy score.")
    for f in sorted((ROOT / "test_data").glob("*.csv")):
        st.download_button(f"Download {f.name}", f.read_bytes(), f.name, "text/csv", key=f"td_{f.name}")
    up = st.file_uploader("Upload a CSV", type="csv")
    if up:
        df = pd.read_csv(up, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if not {"record_id", "source_text"} <= set(df.columns):
            st.error("The file needs record_id and source_text columns.")
        else:
            read_out = pd.DataFrame([{"record_id": r["record_id"], **extract_facts(r["source_text"])}
                                     for _, r in df.iterrows()])
            st.subheader("Facts read from each record")
            st.dataframe(read_out, hide_index=True, width="stretch")
            download(read_out, "facts_read.csv")
            if any(c.startswith("expected_") for c in df.columns):
                table, wrong = score_file(df)
                st.subheader("Accuracy against the answer key")
                st.dataframe(table, hide_index=True)
                if len(wrong):
                    st.write("Errors")
                    st.dataframe(wrong, hide_index=True, width="stretch")
