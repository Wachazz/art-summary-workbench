"""Read key ART facts from free text and audit a summary against its source (Section 3.12.2).

Rules only. They cover the patterns tested in tests/test_all.py and will miss other wording,
so a clinician still reads everything the script cannot resolve."""
import re

DATE = r"\d{4}-\d{2}-\d{2}"
REGIMEN = r"\b((?:TDF|AZT|ABC|D4T)(?:/[A-Z0-9]{2,5}){1,3}|TLD|TLE)\b"
ALIASES = {"TLD": "TDF/3TC/DTG", "TLE": "TDF/3TC/EFV"}
FIELDS = ["regimen", "regimen_changed_from", "viral_load", "viral_load_date", "suppression",
          "cd4", "adherence", "tb_screen", "allergy", "comorbidity", "next_appointment"]
LABELS = {"regimen": "Current regimen", "regimen_changed_from": "Previous regimen",
          "viral_load": "Viral load", "viral_load_date": "Viral load date",
          "suppression": "Suppression", "cd4": "CD4", "adherence": "Adherence",
          "tb_screen": "TB screen", "allergy": "Allergy", "comorbidity": "Comorbidity",
          "next_appointment": "Next appointment"}
# Suppression threshold in copies/mL. 1000 follows common programme practice. CONFIRM with the panel.
SUPPRESSED_BELOW = 1000


def visits_oldest_first(text):
    """Split text on [date] stamps and sort by date, so later notes override earlier ones."""
    parts = re.split(rf"\[({DATE})\]", text)
    out = [(None, parts[0])] if parts[0].strip() else []
    out += [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    return sorted(out, key=lambda v: v[0] or "")


def _reg(token):
    return ALIASES.get(token.upper(), token.upper())


def _last(pattern, text, flags=re.I):
    m = list(re.finditer(pattern, text, flags))
    return m[-1] if m else None


def _clean_list(value):
    v = value.strip(" .;,:").lower()
    if v in ("none", "nil", "no", "none known", "nkda", "nka"):
        return "none"
    return v


def read_visit(body):
    f = {}
    changed = _last(rf"(?:changed|switched)\s+from\s+{REGIMEN}\s+to\s+{REGIMEN}", body)
    if changed:
        f["regimen_changed_from"] = _reg(changed.group(1))
        f["regimen"] = _reg(changed.group(2))
    else:
        prev = _last(rf"(?:switched|changed)\s+from\s+{REGIMEN}", body)
        if prev:
            f["regimen_changed_from"] = _reg(prev.group(1))
        current = [m for m in re.finditer(REGIMEN, body, re.I)
                   if not re.search(r"(?:from|previous(?:ly)?)\s*$", body[:m.start()], re.I)]
        if current:
            f["regimen"] = _reg(current[-1].group(1))

    vl = _last(r"(?:viral load|\bVL\b)(?:\s+(?:latest|result))?\s*(?:is|of|was|:|=)?\s*"
               r"(not\s+suppressed|not\s+detected|undetectable|suppressed|(?:<\s*)?\d[\d,]*(?:\s\d{3})*)", body)
    if vl:
        v = re.sub(r"\s+", " ", vl.group(1).lower())
        if v in ("not suppressed", "suppressed"):
            f["suppression"] = v
        elif v in ("not detected", "undetectable"):
            f["viral_load"], f["suppression"] = "not detected", "suppressed"
        else:
            num = re.sub(r"[^\d<]", "", v)
            f["viral_load"] = num
            f["suppression"] = "suppressed" if int(num.lstrip("<")) < SUPPRESSED_BELOW or (
                num.startswith("<") and int(num[1:]) <= SUPPRESSED_BELOW) else "not suppressed"

    dated = _last(rf"(?:viral load|\bVL\b)[^.;\[]*?\bon\s+({DATE})", body)
    if dated:
        f["viral_load_date"] = dated.group(1)

    m = _last(r"\bCD4\b(?:\s+count)?\s*(?:is|of|was|:|=)?\s*(\d{1,4})\b", body)
    if m:
        f["cd4"] = m.group(1)
    m = _last(r"adherence\s*(?:is|was|:|=)?\s*(good|fair|poor)|\b(good|fair|poor)\s+adherence", body)
    if m:
        f["adherence"] = (m.group(1) or m.group(2)).lower()
    m = _last(r"\bTB\s+screen(?:ing)?\s*(?:is|was|:|=)?\s*(?:was\s+)?(negative|positive)", body)
    if m:
        f["tb_screen"] = m.group(1).lower()

    if re.search(r"\b(?:no known (?:drug )?allerg\w*|nil allerg\w*|no (?:drug )?allerg\w*|NKDA|NKA)\b", body, re.I):
        f["allergy"] = "none"
    else:
        m = (_last(r"allerg(?:y|ies)\s*(?::|=)\s*([^.;\n]+)", body)
             or _last(r"\ballerg(?:y|ies)\s+(none|nil|cotrimoxazole rash|penicillin allergy)", body)
             or _last(r"\ballergic to\s+([^.;\n]+)", body)
             or _last(r"\b((?:penicillin|sulpha|sulfa) allergy|cotrimoxazole rash)\b", body))
        if m:
            f["allergy"] = _clean_list(m.group(1))

    if re.search(r"\bno comorbid\w*", body, re.I):
        f["comorbidity"] = "none"
    else:
        m = (_last(r"comorbidit(?:y|ies)\s*(?::|=)?\s*([^.;\n]+)", body)
             or _last(r"\balso has\s+([^.;\n]+)", body))
        if m:
            f["comorbidity"] = _clean_list(m.group(1))

    m = _last(rf"(?:next appointment|review again|next visit|review)\s*(?:is|on|:|=)?\s*(?:on\s+)?({DATE})", body)
    if m:
        f["next_appointment"] = m.group(1)
    return f


def extract_facts(text):
    """Latest value of every field. None means the rules found nothing."""
    facts = {k: None for k in FIELDS}
    for date, body in visits_oldest_first(text):
        found = read_visit(body)
        if "viral_load" in found or "suppression" in found:
            # a newer viral load result replaces the older value, suppression and date together
            facts["viral_load"] = facts["suppression"] = None
            facts["viral_load_date"] = date
        facts.update(found)
    return facts


def _same(field, a, b):
    norm = lambda x: re.sub(r"\s+", " ", str(x).lower().strip(" ."))
    a, b = norm(a), norm(b)
    if field == "allergy":
        strip = lambda x: re.sub(r"\b(allergy|rash)\b", "", x).strip()
        return strip(a) == strip(b)
    return a == b


def audit(summary, source):
    """Label each field correct, wrong (intrinsic), missing, or unsupported (extrinsic)."""
    src, out = extract_facts(source), extract_facts(summary)
    rows = []
    for k in FIELDS:
        s, o = src[k], out[k]
        if s is None and o is None:
            continue
        if s is None:
            label = "unsupported"
        elif o is None:
            label = "missing"
        else:
            label = "correct" if _same(k, s, o) else "wrong"
        # "none" recorded for allergy or comorbidity and left out of a summary is not an omission
        if label == "missing" and k in ("allergy", "comorbidity") and s == "none":
            continue
        rows.append({"field": k, "source": s or "", "summary": o or "", "result": label})
    return rows
