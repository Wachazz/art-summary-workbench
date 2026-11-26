"""Compare facts read from a file of notes with its expected_ columns."""
import pandas as pd
from facts import _same, extract_facts

def score_file(df):
    out, wrong = {}, []
    keys = [c[9:] for c in df.columns if c.startswith("expected_")]
    for _, r in df.iterrows():
        got = extract_facts(str(r["source_text"]))
        for k in keys:
            want, have = str(r["expected_" + k]), got.get(k)
            ok = have is not None and _same(k, want, have)
            out.setdefault(k, []).append(ok)
            if not ok:
                wrong.append({"record_id": r["record_id"], "field": k, "expected": want, "read": have or "not found"})
    table = pd.DataFrame([{"field": k, "correct": sum(v), "of": len(v), "accuracy": round(sum(v) / len(v), 3)}
                          for k, v in out.items()])
    return table, pd.DataFrame(wrong)

if __name__ == "__main__":
    import sys
    t, w = score_file(pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False))
    print(t.to_string(index=False)); print(w.head(25).to_string(index=False) if len(w) else "no errors")
