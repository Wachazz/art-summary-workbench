"""Study calculations fixed in Chapter 3: I-CVI, crossover plan, hypothesis tests, kappa
and the four acceptance criteria. Nothing here creates data. Empty input gives 'Pending'."""
import random
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ Objective 1 (3.8)
# key: (label, examples, pattern used for document frequency, safety critical)
FEATURES = {
    "treatment": ("Treatment history", "Regimen, start date, changes and reasons, line of therapy",
                  r"\b(TDF|AZT|ABC|3TC|DTG|EFV|TLD)\b|regimen", False),
    "virological": ("Virological status", "Latest viral load and date, trend, suppression",
                    r"viral load|\bVL\b|suppress", False),
    "immunological": ("Immunological and clinical status", "CD4 count, WHO stage",
                      r"\bCD4\b|WHO stage", False),
    "tb": ("Tuberculosis status", "TB screen, TB treatment, preventive therapy", r"\bTB\b|\bTPT\b", True),
    "coinfection": ("Other co-infection and comorbidity", "Cotrimoxazole, infections, chronic conditions",
                    r"cotrimoxazole|opportunistic|comorbidity: (?!none)|hypertension|diabetes|asthma", False),
    "allergy": ("Allergy", "Drug allergy. Added from the prototype's coded fields",
                r"allerg(?!ies: none)", False),
    "adherence": ("Adherence and retention", "Missed visits, counselling, interruption",
                  r"adheren|missed|\bEAC\b|interrupt", False),
    "mental": ("Mental health and substance use", "Depression, alcohol, other substances",
               r"depress|alcohol|substance|mental", False),
    "social": ("Social circumstances and stigma", "Disclosure, supporter, food, mobility, stigma",
               r"disclos|stigma|supporter|food insecur|travel|transport", False),
    "pregnancy": ("Pregnancy and PMTCT", "Pregnancy status, infant prophylaxis, breastfeeding",
                  r"pregnan|breastfeed|\bPMTCT\b|infant prophylaxis", True),
    "pending": ("Pending actions", "Next viral load due, pending results, referrals",
                r"next appointment|review again|referr|pending|\bdue\b", False),
}
ICVI_CUT, FREQ_CUT, MIN_PANEL, PANEL_TARGET = 0.78, 0.05, 6, 8


def document_frequency(texts):
    return {k: (sum(bool(re.search(v[2], t, re.I)) for t in texts) / len(texts) if texts else 0.0)
            for k, v in FEATURES.items()}


def feature_decisions(ratings, doc_freq):
    """ratings: DataFrame, one row per clinician, one column per feature, values 1 to 4."""
    n = len(ratings)
    rows = []
    for k, (label, _, _, safety) in FEATURES.items():
        agree = int((ratings[k] >= 3).sum()) if n else 0
        icvi = agree / n if n else None
        if n < MIN_PANEL:
            decision = f"Pending ({n} of at least {MIN_PANEL} raters)"
        elif safety:
            decision = "Retained (safety critical)"
        elif icvi >= ICVI_CUT and doc_freq[k] >= FREQ_CUT:
            decision = "Retained"
        else:
            decision = "Not retained"
        rows.append({"key": k, "Feature": label, "Raters 3 or 4": f"{agree} of {n}",
                     "I-CVI": None if icvi is None else round(icvi, 2),
                     "In records": round(doc_freq[k], 3), "Decision": decision})
    return pd.DataFrame(rows).sort_values("I-CVI", ascending=False, na_position="last")


# ------------------------------------------------------------------ crossover (3.12.4)
SEQUENCES = {1: [("Manual", "A"), ("Assisted", "B")], 2: [("Manual", "B"), ("Assisted", "A")],
             3: [("Assisted", "A"), ("Manual", "B")], 4: [("Assisted", "B"), ("Manual", "A")]}


def allocate(n=15, seed=2026):
    """Block randomisation in fours, so sequence sizes differ by at most one."""
    rng, order = random.Random(seed), []
    while len(order) < n:
        block = [1, 2, 3, 4]
        rng.shuffle(block)
        order += block
    return {f"C{i + 1:02d}": order[i] for i in range(n)}


QUESTIONS = [
    "What is the current regimen?",
    "What are the date and result of the latest viral load?",
    "Is the patient virally suppressed?",
    "Was the regimen changed in the previous 12 months? If so, how?",
    "Is there any active opportunistic infection or TB treatment?",
    "Is any adherence concern recorded?",
    "Is any disclosure, stigma or mental health concern recorded?",
    "What action is due at the next visit?",
]


def mixed_model(df, outcome):
    """outcome ~ condition + period, random intercepts for clinician and record (crossed).
    Returns (assisted minus manual, low, high, p)."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["assisted"] = (d["condition"] == "Assisted").astype(int)
    d["y"] = np.log(d["seconds"]) if outcome == "log_seconds" else d["score"]
    d["one"] = 1
    model = smf.mixedlm("y ~ assisted + C(period)", d, groups="one", re_formula="0",
                        vc_formula={"clinician": "0 + C(clinician)", "record": "0 + C(record)"})
    for method in ("lbfgs", "bfgs", "powell", "cg"):
        fit = model.fit(reml=True, method=method, maxiter=1000)
        if fit.converged:
            break
    est = float(fit.params["assisted"])
    low, high = (float(v) for v in fit.conf_int().loc["assisted"])
    if not fit.converged:
        raise ValueError("the mixed model did not converge, analyse the CSV in R (lme4)")
    return est, low, high, float(fit.pvalues["assisted"])


def crossover_results(df):
    """df columns: clinician, record, condition, period, seconds, score (0 to 8, may be blank)."""
    out = {"note": "", "time": None, "comprehension": None}
    if df.empty or df["condition"].nunique() < 2 or df["clinician"].nunique() < 3:
        out["note"] = "Pending. Needs both conditions and at least 3 clinicians to fit."
        return out
    n = df["clinician"].nunique()
    if n < 12:
        out["note"] = f"Pilot only: {n} clinicians, below the minimum of 12. No formal hypothesis decision."
    est, low, high, p = mixed_model(df, "log_seconds")
    out["time"] = {"ratio": float(np.exp(est)), "low": float(np.exp(low)), "high": float(np.exp(high)), "p": p}
    scored = df.dropna(subset=["score"])
    if len(scored) == len(df):
        est, low, high, _ = mixed_model(scored, "score")
        out["comprehension"] = {"difference": est, "low": low, "high": high, "non_inferior": low > -1}
    else:
        out["note"] += f" Comprehension pending: {len(df) - len(scored)} reviews not yet scored."
    return out


def paired_checks(df):
    """Wilcoxon signed-rank on clinician means and Cohen's dz."""
    from scipy.stats import wilcoxon
    m = df.groupby(["clinician", "condition"])["seconds"].mean().unstack().dropna()
    if len(m) < 3:
        return None
    diff = m["Assisted"] - m["Manual"]
    return {"p": float(wilcoxon(diff).pvalue), "dz": float(diff.mean() / diff.std(ddof=1)), "n": len(m)}


def weighted_kappa(ratings, column):
    """Items rated by exactly two raters. Quadratic weights."""
    from sklearn.metrics import cohen_kappa_score
    pairs = [g.sort_values("rater")[column].tolist()
             for _, g in ratings.groupby(["record", "system"]) if g["rater"].nunique() == 2 and len(g) == 2]
    if len(pairs) < 2:
        return None
    a, b = zip(*pairs)
    k = cohen_kappa_score(a, b, weights="quadratic", labels=list(range(0, 6)))
    return None if np.isnan(k) else float(k)


# ------------------------------------------------------------------ acceptance (Table 3.9)
def acceptance(a=None, b=None, c=None, d=None):
    """a: (rougeL diff low, BERTScore diff low). b: (regimen acc, viral load acc, serious harm count).
    c: (correctness mean, completeness mean). d: (time high ratio, time p, comprehension low)."""
    def row(name, rule, value, passed):
        return {"Criterion": name, "Threshold": rule,
                "Result": "Pending" if value is None else value,
                "Met": "Pending" if value is None else ("Yes" if passed else "No")}
    return pd.DataFrame([
        row("A. Beats best baseline", "95% interval of ROUGE-L and BERTScore differences above zero",
            None if a is None else f"lower limits {a[0]:.3f}, {a[1]:.3f}", a is not None and min(a) > 0),
        row("B. Safe on key fields", "Regimen and viral load accuracy 95% or higher, no serious harm",
            None if b is None else f"{b[0]:.1%}, {b[1]:.1%}, {b[2]} serious",
            b is not None and b[0] >= .95 and b[1] >= .95 and b[2] == 0),
        row("C. Clinicians rate it usable", "Correctness 4.0 or higher, completeness 3.5 or higher",
            None if c is None else f"{c[0]:.2f}, {c[1]:.2f}", c is not None and c[0] >= 4 and c[1] >= 3.5),
        row("D. Saves time, keeps facts", "Faster at 5% level, comprehension lower limit above -1",
            None if d is None else f"ratio upper {d[0]:.2f}, p {d[1]:.3f}, comprehension lower {d[2]:.2f}",
            d is not None and d[0] < 1 and d[1] < .05 and d[2] > -1),
    ])
