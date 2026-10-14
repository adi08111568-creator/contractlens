"""Rules-based baseline: a stand-in for AArete's 2020-2023 'positional rules' phase.

How it was written (so the comparison is fair): regex rules were written for the two
layouts a team would have seen first - A (formal articles) and B (rate-schedule tables) -
plus a few generic first-match fallbacks. Nothing was tuned to fail on the other layouts.
First match in document order wins, as in a simple rules engine; there is no notion of
amendments, superseded terms or look-alike numbers. Accuracy is whatever it measures.
"""
import re

from agent.doc import collapse, date_spans
from agent.tools import FIELD_NAMES

_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming",
]
_STATES.sort(key=len, reverse=True)   # longest first so 'West Virginia' beats 'Virginia'
STATE_RE = "|".join(re.escape(s) for s in _STATES)


def _first(text, patterns, flags=re.I):
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m
    return None


def _num(m):
    return int(m.group(1).replace(",", "")) if m else None


def _date_after(text, anchor_patterns, window=60):
    spans = date_spans(text)
    for p in anchor_patterns:
        m = re.search(p, text, re.I)
        if m:
            for pos, iso in spans:
                if m.end() <= pos <= m.end() + window:
                    return iso
    return None


def extract(raw_text: str) -> dict:
    t = collapse(raw_text)
    out = {}

    m = _first(t, [r'between (.+?) \("Plan"\) and (.+?) \("Provider"\)',
                   r"Payer\W+(.+?)\W+Provider\W+(.+?)\W+Effective Date",
                   r"between (.+?) and (.+?)\."])
    out["payer_name"] = m.group(1).strip() if m else None
    out["provider_name"] = m.group(2).strip() if m else None

    out["effective_date"] = _date_after(t, [r"entered into as of", r"Effective Date\W", r"made on"])
    out["expiry_date"] = _date_after(t, [r"expiring on", r"Expiration Date\W", r"ends on( the)?", r"expires on"])

    if _first(t, [r"shall automatically renew", r"Evergreen"]):
        out["auto_renewal"] = True
    elif _first(t, [r"not renew automatically", r"no automatic renewal", r"Fixed term"]):
        out["auto_renewal"] = False
    else:
        out["auto_renewal"] = None

    out["termination_notice_days"] = _num(_first(t, [r"without cause upon (\d+) days",
                                                     r"Termination Without Cause\W+(\d+) days"]))
    out["inpatient_per_diem_usd"] = _num(_first(t, [r"Acute[^$]{0,90}\$([\d,]+)", r"per diem[^$]{0,40}\$([\d,]+)"]))
    out["outpatient_pct_of_medicare"] = _num(_first(t, [r"Outpatient - General.{0,60}?(\d+)%",
                                                        r"(\d+)% of (?:the )?(?:then-current )?(?:Medicare|MCR)"]))
    out["clean_claim_payment_days"] = _num(_first(t, [r"clean claim within (\d+) days",
                                                      r"Clean claim payment\W+Within (\d+)"]))
    out["timely_filing_days"] = _num(_first(t, [r"within (\d+) days after the date of service",
                                                r"Timely filing\W+(\d+) days"]))
    out["credentialing_sla_days"] = _num(_first(t, [r"within (\d+) days after receipt of a complete application",
                                                    r"Credentialing turnaround\W+(\d+) days"]))
    m = _first(t, [rf"State of ({STATE_RE})\b"])
    out["governing_law"] = m.group(1) if m else None
    return {k: out.get(k) for k in FIELD_NAMES}
