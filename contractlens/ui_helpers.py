"""Presentation helpers for the Streamlit app (kept out of app.py so app.py reads top-to-bottom)."""
import io
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agent.doc import ContractDoc
from agent.tools import FIELD_NAMES
from evaluate import equal

LAYOUT_LABELS = {
    "A_formal_articles": "A · Formal articles",
    "B_rate_tables": "B · Rate tables",
    "C_plain_letter": "C · Plain letter",
    "D_definitions_amendment": "D · With amendment",
    "E_term_sheet": "E · Term sheet",
}
FIELD_LABELS = {
    "payer_name": "Payer", "provider_name": "Provider", "effective_date": "Effective date",
    "expiry_date": "Expiry date", "auto_renewal": "Auto-renewal",
    "termination_notice_days": "Termination notice (days)", "inpatient_per_diem_usd": "Inpatient per diem ($/day)",
    "outpatient_pct_of_medicare": "Outpatient (% of Medicare)", "clean_claim_payment_days": "Clean-claim payment (days)",
    "timely_filing_days": "Timely filing (days)", "credentialing_sla_days": "Credentialing SLA (days)",
    "governing_law": "Governing law",
}

# Validated with the dataviz skill's validate_palette.js (categorical slots 1-2): ALL CHECKS PASS in both modes.
PALETTE = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e3e2de",
              "rules": "#2a78d6", "agent": "#eb6834"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#34332f",
             "rules": "#3987e5", "agent": "#d95926"},
}


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - older Streamlit / headless test runner
        return "light"


# ---------------------------------------------------------------- data loading
def doc_from_pdf_bytes(data: bytes, name: str) -> ContractDoc:
    from pypdf import PdfReader
    text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return ContractDoc(text, name=name)


def cache_signature(cache_dir: Path):
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(cache_dir.glob("*.json"))) if cache_dir.exists() else ()


@st.cache_data(show_spinner=False)
def load_cache(sig, cache_dir: str) -> dict:
    """{contract_id: saved agent result}. `sig` only exists to invalidate the cache when files change."""
    out = {}
    for p in sorted(Path(cache_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
            out[r["contract"]] = r
        except (ValueError, KeyError):
            continue
    return out


def save_run(cache_dir: Path, result: dict, model: str):
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{result['contract']}.json").write_text(json.dumps({**result, "model": model}, indent=2, default=str))
        return True
    except OSError:
        return False  # read-only filesystem (e.g. some hosts) - the run still shows in the session


# ---------------------------------------------------------------- formatting
def fmt_value(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def fields_dataframe(result: dict, truth: dict | None = None) -> pd.DataFrame:
    rows = []
    for f in FIELD_NAMES:
        v = result["fields"][f]
        row = {"Field": FIELD_LABELS[f], "Value": fmt_value(v["value"]),
               "Status": "✅ auto" if v["status"] == "auto" else "👀 review",
               "Confidence": float(v["confidence"]), "Evidence (quote from contract)": v["quote"], "Note": v["reason"]}
        if truth is not None:
            ok = equal(f, v["value"], truth[f])
            row["Answer key"] = "✔ correct" if ok else f"✘ expected {fmt_value(truth[f])}"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_call(call: dict) -> str:
    t, a, r = call["tool"], call.get("args") or {}, call.get("result") or {}
    if isinstance(r, dict) and r.get("error") and t != "finish":
        return f"⚠️ `{t}` → {r['error']}"
    if t == "search_contract":
        ids = [m["chunk_id"] for m in r.get("matches", [])]
        return f"🔎 searched “{a.get('query', '')}” → chunks {ids or 'none'}"
    if t == "read_chunk":
        return f"📖 read chunk {a.get('chunk_id')}"
    if t == "submit_fields":
        n_items = len(a.get("fields") or [])
        issues = r.get("issues", [])
        head = f"📝 submitted {n_items} field(s) → ✅ {len(r.get('verified', []))} verified"
        if issues:
            head += f", ⚠️ {len(issues)} issue(s) sent back for correction"
        return head + "".join(f"\n  - `{i.get('field')}`: {i.get('problem')}" for i in issues)
    if t == "flag_for_human":
        return f"🚩 flagged `{a.get('field')}` for a human — {a.get('reason', '')}"
    if t == "finish":
        return "🏁 finish → done" if r.get("status") == "done" else f"🏁 finish refused — unresolved: {r.get('unresolved')}"
    return f"`{t}`"


def render_step(entry: dict):
    head = f"**Step {entry['step']}**"
    if entry.get("thought"):
        head += f" — {entry['thought']}"
    st.markdown(head)
    for c in entry.get("calls", []):
        st.markdown(summarize_call(c))


def silent_errors_per_contract(runs: dict, key: dict):
    """Average number of auto-accepted fields that were wrong (only meaningful for real runs on sample contracts)."""
    counts = []
    for cid, r in runs.items():
        if cid not in key:
            continue
        counts.append(sum(1 for f, v in r["fields"].items()
                          if v["status"] == "auto" and not equal(f, v["value"], key[cid]["truth"][f])))
    return sum(counts) / len(counts) if counts else None


# ---------------------------------------------------------------- charts
def _style(chart, p):
    return (chart.configure_view(stroke=None).configure(background="transparent")
            .configure_axis(labelColor=p["muted"], titleColor=p["muted"], domainColor=p["grid"], tickColor=p["grid"])
            .configure_legend(labelColor=p["text"], titleColor=p["muted"]))


def accuracy_chart(df: pd.DataFrame, mode: str):
    """Grouped bars: field accuracy by contract layout, one bar per method. Single hue per method (fixed order)."""
    p = PALETTE[mode]
    methods = [m for m in ("Rules baseline", "Agent") if m in set(df["Method"])]
    colors = {"Rules baseline": p["rules"], "Agent": p["agent"]}
    order = list(dict.fromkeys(df["Layout"]))
    base = alt.Chart(df).encode(
        x=alt.X("Layout:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, ticks=False, labelLimit=150)),
        xOffset=alt.XOffset("Method:N", sort=methods))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=28, stroke=p["surface"], strokeWidth=2).encode(
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100]), title="Field accuracy (%)",
                axis=alt.Axis(grid=True, gridColor=p["grid"], domain=False, ticks=False, tickCount=5)),
        color=alt.Color("Method:N", scale=alt.Scale(domain=methods, range=[colors[m] for m in methods]),
                        legend=alt.Legend(title=None, orient="top", symbolType="square")),
        tooltip=[alt.Tooltip("Layout:N"), alt.Tooltip("Method:N"), alt.Tooltip("Accuracy:Q", format=".1f", title="Accuracy %")])
    labels = base.mark_text(dy=-8, fontSize=12, color=p["text"]).encode(
        y="Accuracy:Q", text=alt.Text("Accuracy:Q", format=".0f"))
    return _style((bars + labels).properties(height=320), p)


def cost_chart(manual: float, ai: float, mode: str):
    p = PALETTE[mode]
    df = pd.DataFrame({"Approach": ["Manual analysts", "With agent"], "Annual cost ($)": [manual, ai]})
    base = alt.Chart(df).encode(y=alt.Y("Approach:N", sort=None, title=None, axis=alt.Axis(ticks=False, labelLimit=160)))
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=34).encode(
        x=alt.X("Annual cost ($):Q", title=None, axis=alt.Axis(format="$,.0f", grid=True, gridColor=p["grid"],
                                                              domain=False, ticks=False, tickCount=4)),
        color=alt.value(p["rules"]),
        tooltip=[alt.Tooltip("Approach:N"), alt.Tooltip("Annual cost ($):Q", format="$,.0f")])
    labels = base.mark_text(align="left", dx=6, fontSize=13, color=p["text"]).encode(
        x="Annual cost ($):Q", text=alt.Text("Annual cost ($):Q", format="$,.0f"))
    return _style((bars + labels).properties(height=150), p)
