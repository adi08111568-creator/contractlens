"""The agent's tools, plus the validator that checks its answers against the contract.

Design idea: the LLM proposes values, but *the environment verifies them*. Every field
must come with a verbatim quote. Code then checks that (1) the quote really exists in the
contract, (2) the value is supported by the quote, (3) the value is in a plausible range,
and (4) cross-field logic holds. Failed checks go back to the agent as tool results, so it
must re-read the contract and fix its answer - this is the self-correction loop.
"""
import re
from datetime import date

from .doc import ContractDoc, collapse, dates_in, number_supported, norm

FIELD_SPECS = {
    "payer_name": dict(kind="str", anchor=None,
                       desc="Legal name of the health plan / payer that pays claims. Name only, no role label."),
    "provider_name": dict(kind="str", anchor=None,
                          desc="Legal name of the hospital / provider / facility. Name only, no role label."),
    "effective_date": dict(kind="date", anchor=r"effective|eff\.|begin|commenc|made on|entered|date",
                           desc="Date the agreement takes effect, as YYYY-MM-DD."),
    "expiry_date": dict(kind="date", anchor=r"expir|exp\.|\bend|\bterm",
                        desc="Last day of the initial term, as YYYY-MM-DD."),
    "auto_renewal": dict(kind="bool", anchor=r"renew|roll over|evergreen|auto",
                         desc="true if the agreement renews automatically after the initial term, else false."),
    "termination_notice_days": dict(kind="int", anchor=r"terminat|without cause|end it early|notice",
                                    desc="Days of written notice to terminate WITHOUT cause (not non-renewal notice)."),
    "inpatient_per_diem_usd": dict(kind="int", anchor=r"per diem|per day|/ day|inpatient|ip ",
                                   desc="Current per-day dollar rate for ACUTE medical/surgical inpatient services "
                                        "(not behavioral health). Integer dollars."),
    "outpatient_pct_of_medicare": dict(kind="int", anchor=r"medicare|outpatient|mcr|op ",
                                       desc="Current percent of the Medicare fee schedule for GENERAL outpatient "
                                            "services (not emergency department). Integer."),
    "clean_claim_payment_days": dict(kind="int", anchor=r"clean claim|ccp|claim",
                                     desc="Days within which the plan must pay a clean claim after receipt."),
    "timely_filing_days": dict(kind="int", anchor=r"timely|filing|submit|send claims|date of service|dos",
                               desc="Days after the date of service within which the provider must submit claims "
                                    "(not the appeals window)."),
    "credentialing_sla_days": dict(kind="int", anchor=r"credential|cred\.",
                                   desc="Days within which the plan must complete credentialing."),
    "governing_law": dict(kind="str", anchor=r"law|gov",
                          desc="US state whose law governs the agreement. State name only, e.g. 'Texas'."),
}
FIELD_NAMES = list(FIELD_SPECS)

RANGES = {
    "inpatient_per_diem_usd": (500, 10000),
    "outpatient_pct_of_medicare": (50, 300),
    "clean_claim_payment_days": (1, 120),
    "timely_filing_days": (30, 730),
    "termination_notice_days": (1, 365),
    "credentialing_sla_days": (1, 365),
}
MAX_QUOTE_CHARS = 500
MIN_QUOTE_CHARS = 8

TOOL_SPECS = [
    {
        "name": "search_contract",
        "description": "Keyword search over the contract. Returns up to 4 chunks (with ids) that best match "
                       "the words in the query. Use specific words, e.g. 'per diem inpatient' or 'amendment'.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Keywords to look for."}},
                       "required": ["query"]},
    },
    {
        "name": "read_chunk",
        "description": "Read one chunk of the contract in full by its id (ids are listed in the outline).",
        "parameters": {"type": "object",
                       "properties": {"chunk_id": {"type": "integer"}},
                       "required": ["chunk_id"]},
    },
    {
        "name": "submit_fields",
        "description": "Submit extracted values. Each item needs the field name, the value (as text), a VERBATIM "
                       "quote copied from the contract that supports it, and your confidence 0-1. The tool "
                       "verifies every quote and value and returns issues to fix. You may call it repeatedly; "
                       "later submissions replace earlier ones for the same field.",
        "parameters": {"type": "object", "properties": {"fields": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "field": {"type": "string", "enum": FIELD_NAMES},
                "value": {"type": "string", "description": "Dates as YYYY-MM-DD, booleans as true/false, "
                                                           "numbers as digits only."},
                "quote": {"type": "string", "description": "Verbatim text from the contract (max 500 chars). For table values include the row label."},
                "confidence": {"type": "number", "description": "0 to 1"}},
                "required": ["field", "value", "quote"]}}},
            "required": ["fields"]},
    },
    {
        "name": "flag_for_human",
        "description": "Send a field to the human review queue when the contract is ambiguous, contradictory or "
                       "silent about it. Prefer this to guessing.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "enum": FIELD_NAMES},
            "reason": {"type": "string"}},
            "required": ["field", "reason"]},
    },
    {
        "name": "finish",
        "description": "Call when every field is either verified or flagged. Refused if fields are unresolved.",
        "parameters": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------- coercion / checks
def coerce(field: str, raw):
    """Return (value, error) for the raw text the model gave us."""
    kind = FIELD_SPECS[field]["kind"]
    s = str(raw).strip()
    if kind == "str":
        return (s, None) if s else (None, "empty value")
    if kind == "bool":
        if s.lower() in ("true", "yes", "y", "1"):
            return True, None
        if s.lower() in ("false", "no", "n", "0"):
            return False, None
        return None, "boolean must be true or false"
    if kind == "int":
        m = re.search(r"\d[\d,]*", s)
        if not m:
            return None, "number must be digits, e.g. 90"
        return int(m.group().replace(",", "")), None
    if kind == "date":
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            try:
                date.fromisoformat(s)
                return s, None
            except ValueError:
                return None, "not a real calendar date"
        found = sorted(dates_in(s))
        return (found[0], None) if found else (None, "date must be YYYY-MM-DD")
    return None, "unknown field kind"


def check_field(doc: ContractDoc, field: str, value, quote: str):
    """Return a list of problems (empty list = verified)."""
    problems = []
    q = (quote or "").strip()
    if len(norm(q)) < MIN_QUOTE_CHARS:
        return ["quote is missing or too short - copy the exact sentence or table row from the contract"]
    if len(q) > MAX_QUOTE_CHARS:
        problems.append(f"quote is longer than {MAX_QUOTE_CHARS} characters - quote only the key sentence")
    if not doc.contains_quote(q):
        problems.append("quote was not found in the contract - copy the text exactly (no paraphrasing)")
        return problems
    spec = FIELD_SPECS[field]
    kind = spec["kind"]
    if kind == "date" and value not in dates_in(q):
        problems.append(f"the quote does not contain the date {value}")
    elif kind == "int" and not number_supported(value, q):
        problems.append(f"the quote does not contain the number {value} (digits or words)")
    elif kind == "str" and norm(str(value)) not in norm(q):
        problems.append(f"the quote does not contain the text '{value}'")
    if spec["anchor"] and not re.search(spec["anchor"], q, re.I):
        problems.append(f"the quote does not seem to address {field} - find the clause that does")
    if field in RANGES and isinstance(value, int):
        lo, hi = RANGES[field]
        if not lo <= value <= hi:
            problems.append(f"{value} is outside the plausible range {lo}-{hi} for {field}")
    return problems


class Session:
    """Holds one contract's extraction state and executes the agent's tool calls."""

    def __init__(self, doc: ContractDoc):
        self.doc = doc
        self.fields = {}   # name -> {value, quote, confidence, ok, issues}
        self.flags = {}    # name -> reason
        self.finished = False

    # ---- tools
    def search_contract(self, query: str):
        hits = self.doc.search(query)
        return {"matches": [{"chunk_id": c["id"], "text": c["text"]} for c in hits]} if hits \
            else {"matches": [], "note": "no chunk matched; try other keywords or read chunks from the outline"}

    def read_chunk(self, chunk_id):
        c = self.doc.get(int(chunk_id))
        return {"chunk_id": c["id"], "text": c["text"]} if c else {"error": f"no chunk with id {chunk_id}"}

    def submit_fields(self, fields):
        if not isinstance(fields, list) or not fields:
            return {"error": "fields must be a non-empty list of {field, value, quote, confidence}"}
        issues, accepted = [], []
        for item in fields:
            name = (item or {}).get("field")
            if name not in FIELD_SPECS:
                issues.append({"field": name, "problem": f"unknown field; use one of {FIELD_NAMES}"})
                continue
            value, err = coerce(name, item.get("value"))
            quote = item.get("quote", "")
            try:
                conf = max(0.0, min(1.0, float(item.get("confidence", 0.8))))
            except (TypeError, ValueError):
                conf = 0.8
            problems = [err] if err else check_field(self.doc, name, value, quote)
            self.fields[name] = {"value": value, "quote": quote.strip(), "confidence": conf,
                                 "ok": not problems, "issues": problems}
            if problems:
                issues += [{"field": name, "problem": p} for p in problems]
            else:
                accepted.append(name)
                self.flags.pop(name, None)
        issues += self._cross_checks()
        return {"verified": accepted, "issues": issues, "missing": self.missing(),
                "flagged": sorted(self.flags),
                "next": "fix every issue by re-reading the contract, then resubmit those fields"
                        if issues or self.missing() else "all fields verified - call finish"}

    def flag_for_human(self, field: str, reason: str):
        if field not in FIELD_SPECS:
            return {"error": f"unknown field {field}"}
        self.flags[field] = reason or "flagged by agent"
        return {"flagged": field, "unresolved": self.unresolved()}

    def finish(self):
        left = self.unresolved()
        if left:
            return {"error": "cannot finish yet", "unresolved": left,
                    "next": "submit verified values for these fields or flag them for a human"}
        self.finished = True
        return {"status": "done"}

    # ---- bookkeeping
    def _cross_checks(self):
        out = []
        e, x = self.fields.get("effective_date"), self.fields.get("expiry_date")
        if e and x and e["ok"] and x["ok"] and x["value"] <= e["value"]:
            for n in ("effective_date", "expiry_date"):
                self.fields[n]["ok"] = False
                self.fields[n]["issues"].append("expiry date must be after the effective date")
            out.append({"field": "expiry_date", "problem": "expiry date must be after the effective date - "
                                                           "re-check both dates"})
        return out

    def missing(self):
        return [n for n in FIELD_NAMES if n not in self.fields and n not in self.flags]

    def unresolved(self):
        return [n for n in FIELD_NAMES
                if n not in self.flags and not (n in self.fields and self.fields[n]["ok"])]

    def call(self, name: str, args: dict):
        fn = {"search_contract": self.search_contract, "read_chunk": self.read_chunk,
              "submit_fields": self.submit_fields, "flag_for_human": self.flag_for_human,
              "finish": self.finish}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # never let a tool crash the loop
            return {"error": f"{type(e).__name__}: {e}"}

    def result(self, threshold: float = 0.75):
        """Final per-field output with an auto / review routing decision."""
        out = {}
        for n in FIELD_NAMES:
            f = self.fields.get(n)
            if n in self.flags:
                out[n] = {"value": f["value"] if f and f["ok"] else None, "quote": f["quote"] if f else "",
                          "confidence": f["confidence"] if f else 0.0, "status": "review",
                          "reason": f"flagged by agent: {self.flags[n]}"}
            elif f and f["ok"] and f["confidence"] >= threshold:
                out[n] = {"value": f["value"], "quote": f["quote"], "confidence": f["confidence"],
                          "status": "auto", "reason": "quote verified"}
            elif f and f["ok"]:
                out[n] = {"value": f["value"], "quote": f["quote"], "confidence": f["confidence"],
                          "status": "review", "reason": f"low confidence ({f['confidence']:.2f} < {threshold})"}
            elif f:
                out[n] = {"value": None, "quote": f["quote"], "confidence": f["confidence"],
                          "status": "review", "reason": "failed verification: " + "; ".join(f["issues"])}
            else:
                out[n] = {"value": None, "quote": "", "confidence": 0.0, "status": "review",
                          "reason": "agent did not extract this field"}
        return out
