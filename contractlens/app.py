"""ContractLens - Streamlit app. Run:  streamlit run app.py"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import baseline_rules  # noqa: F401  (used through evaluate.rules_preds)
from agent.agent import run_agent
from agent.doc import load_contract
from agent.llm import DEFAULT_GEMINI_MODEL, GeminiClient
from agent.mock_llm import MockLLM
from evaluate import agent_routing, rules_preds, score
from impact import ImpactInputs, compute, review_share_from_results, sensitivity
from ui_helpers import (FIELD_LABELS, LAYOUT_LABELS, accuracy_chart, cache_signature, cost_chart, doc_from_pdf_bytes,
                        fields_dataframe, load_cache, render_step, save_run, silent_errors_per_contract, theme_mode)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE_DIR = DATA / "cache" / "agent"
CASE_URL = ("https://www.zenml.io/llmops-database/document-metadata-extraction-at-scale-using-"
            "generative-ai-for-healthcare-and-financial-services")

MODES = {"Replay saved run (real Gemini output)": "replay", "Live agent (Gemini)": "live",
         "Scripted simulation (no AI - interface test)": "sim"}
MODE_NAME = {"replay": "saved run of the agent", "live": "live agent run", "sim": "scripted simulation"}
SIM_WARNING = ("**Scripted simulation - not an AI model.** These answers are produced by a script that knows the "
               "answer key. It exists only to test the interface. Do not present it as model accuracy.")

st.set_page_config(page_title="ContractLens", page_icon="📄", layout="wide")


# ---------------------------------------------------------------- helpers
def secret(name: str):
    try:
        v = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file
        v = None
    return v or os.environ.get(name)


@st.cache_data(show_spinner=False)
def load_key() -> dict:
    return json.loads((DATA / "ground_truth.json").read_text())


@st.cache_data(show_spinner=False)
def rules_results() -> dict:
    return score(rules_preds(DATA, "pdf"), load_key())


KEY = load_key()
SAMPLES = sorted(KEY)
CACHE = load_cache(cache_signature(CACHE_DIR), str(CACHE_DIR))
CACHE = {k: v for k, v in CACHE.items() if k in KEY}
API_KEY = secret("GEMINI_API_KEY") or secret("GOOGLE_API_KEY")
MODE = theme_mode()

if "runs" not in st.session_state:
    st.session_state.runs = {}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    default_idx = 0 if CACHE else (1 if API_KEY else 2)
    mode_label = st.radio("Mode", list(MODES), index=default_idx, key="mode")
    mode = MODES[mode_label]
    if mode == "replay":
        st.caption(f"{len(CACHE)} saved agent run(s) available." if CACHE else
                   "No saved runs yet. Run `python run_agent.py --all`, or use another mode.")
    if mode == "live":
        st.caption("Calls Gemini. Free keys have rate limits." if API_KEY else
                   "No `GEMINI_API_KEY` found. Add it to Streamlit secrets or your environment.")
        model_name = st.text_input("Gemini model", value=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL), key="model")
        if secret("APP_PASSWORD"):
            st.text_input("Access code", type="password", key="access_code")
    if mode in ("live", "sim"):
        threshold = st.slider("Confidence needed for auto-accept", 0.5, 1.0, 0.75, 0.05, key="threshold")
        max_steps = st.slider("Max agent steps", 4, 20, 12, key="max_steps")
    else:
        threshold, max_steps, model_name = 0.75, 12, ""
    if mode == "sim":
        st.warning("Scripted simulation: not an AI. For interface testing only.")
    st.divider()
    st.caption("Synthetic contracts, fictional parties. Coursework demo based on AArete's Doxy AI case study "
               "from the ZenML LLMOps Database.")

# ---------------------------------------------------------------- header
st.title("ContractLens")
st.markdown("An AI agent that reads provider contracts, **proves every value with a quote from the contract**, "
            "fixes its own mistakes, and sends only uncertain fields to a human. "
            f"Based on [AArete's Doxy AI]({CASE_URL}).")

tab_extract, tab_review, tab_acc, tab_impact, tab_how = st.tabs(
    ["Extract", "Review queue", "Accuracy", "Impact", "How it works"])


# ---------------------------------------------------------------- run logic
def run_extraction(doc, name, truth, display):
    """Run one extraction in the selected mode and store the result in session state."""
    res, model_used = None, ""
    if mode == "replay":
        res = CACHE.get(name)
        if res is None:
            st.warning("No saved run for this contract. Run `python run_agent.py` for it, or switch mode.")
            return
        model_used = res.get("model", "")
        animate = st.session_state.get("animate", True)
        with st.status("Replaying the saved agent run…", expanded=True) as status:
            for entry in res["trace"]:
                render_step(entry)
                if animate:
                    time.sleep(0.4)
            status.update(label=f"Replayed {len(res['trace'])} steps", state="complete")
    else:
        if mode == "sim" and truth is None:
            st.warning("Simulation needs a sample contract (it uses the answer key). Pick a sample or switch mode.")
            return
        if mode == "live":
            if not API_KEY:
                st.error("No Gemini API key configured. Add `GEMINI_API_KEY` to Streamlit secrets.")
                return
            if secret("APP_PASSWORD") and st.session_state.get("access_code", "") != secret("APP_PASSWORD"):
                st.error("Enter the access code in the sidebar to use the live agent.")
                return
        try:
            client = GeminiClient(api_key=API_KEY, model=model_name or None) if mode == "live" else MockLLM(doc, truth)
            model_used = client.model if mode == "live" else "scripted-simulation"
            with st.status("Agent working…", expanded=True) as status:
                res = run_agent(doc, client, max_steps=max_steps, threshold=threshold, on_step=render_step)
                status.update(label=f"Finished in {res['steps']} steps ({res['stop_reason']})", state="complete")
        except Exception as e:  # noqa: BLE001 - show a readable error instead of a stack trace
            st.error(f"The agent run failed: {type(e).__name__}: {str(e)[:300]}")
            return
        if mode == "live" and truth is not None:
            save_run(CACHE_DIR, res, model_used)   # real runs only; never the simulation
    st.session_state.runs[display] = {"mode": mode, "result": res, "truth": truth, "model": model_used}


# ---------------------------------------------------------------- Extract tab
with tab_extract:
    src = st.radio("Contract", ["Sample contract", "Upload a PDF"], horizontal=True, key="src")
    doc = truth = None
    display = ""
    if src == "Sample contract":
        name = st.selectbox("Sample", SAMPLES, key="sample",
                            format_func=lambda n: f"{n} · {LAYOUT_LABELS[KEY[n]['layout']]}")
        doc, truth, display = load_contract(DATA / "contracts" / "pdf" / f"{name}.pdf"), KEY[name]["truth"], name
    else:
        up = st.file_uploader("PDF contract (text-based, not scanned)", type=["pdf"], key="upload")
        if up is not None:
            display = f"upload:{up.name}"
            name = Path(up.name).stem
            doc = doc_from_pdf_bytes(up.getvalue(), name)
            if len(doc.text.strip()) < 50:
                st.error("No text found in this PDF. Scanned PDFs need OCR, which this demo does not include.")
                doc = None

    if doc is None:
        st.info("Choose a sample contract or upload a PDF to begin.")
    else:
        with st.expander("View contract text"):
            st.text(doc.text)
        c1, c2 = st.columns([1, 3])
        with c1:
            go = st.button("▶ Run agent", type="primary", key="run", width="stretch")
        with c2:
            if mode == "replay":
                st.toggle("Animate the replay", value=True, key="animate")
        if go:
            run_extraction(doc, name, truth, display)

        shown = st.session_state.runs.get(display)
        if shown:
            res = shown["result"]
            if shown["mode"] == "sim":
                st.warning(SIM_WARNING)
            st.caption(f"Result from a {MODE_NAME[shown['mode']]}"
                       + (f" · model `{shown['model']}`" if shown["model"] else ""))
            check = False
            if shown["truth"] is not None:
                check = st.toggle("Check against the answer key", value=True, key="check")
            m = st.columns(5)
            m[0].metric("Auto-accepted fields", f"{res['auto_fields']} / 12")
            m[1].metric("Sent to human review", res["review_fields"])
            m[2].metric("Issues caught & fixed", res["validation_issues_caught"])
            m[3].metric("Agent steps", res["steps"])
            if check:
                from evaluate import equal
                right = sum(equal(f, v["value"], shown["truth"][f]) for f, v in res["fields"].items())
                m[4].metric("Correct vs answer key", f"{right} / 12")
            else:
                m[4].metric("Tokens used", f"{res['tokens']['prompt'] + res['tokens']['output']:,}")
            st.dataframe(fields_dataframe(res, shown["truth"] if check else None), hide_index=True, width="stretch",
                         column_config={"Field": st.column_config.TextColumn(width=175),
                                        "Value": st.column_config.TextColumn(width=165),
                                        "Status": st.column_config.TextColumn(width=90),
                                        "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", width=105),
                                        "Evidence (quote from contract)": st.column_config.TextColumn(width=250),
                                        "Note": st.column_config.TextColumn(width=140),
                                        "Answer key": st.column_config.TextColumn(width=150)})
            with st.expander(f"Agent reasoning trace ({len(res['trace'])} steps)"):
                for e in res["trace"]:
                    render_step(e)


# ---------------------------------------------------------------- Review queue tab
with tab_review:
    st.subheader("Human review queue")
    st.markdown("Fields the agent could not verify, flagged, or was unsure about. Everything else was verified "
                "against a quote in the contract and accepted automatically.")
    pool = {n: (r, "saved run") for n, r in CACHE.items()}
    for n, s in st.session_state.runs.items():
        pool[n] = (s["result"], MODE_NAME[s["mode"]])
    rows, total_fields = [], 0
    for n, (r, source) in sorted(pool.items()):
        for f, v in r["fields"].items():
            total_fields += 1
            if v["status"] == "review":
                rows.append({"Contract": n, "Field": FIELD_LABELS[f], "Agent value": "—" if v["value"] is None else str(v["value"]),
                             "Decision": "Pending", "Human value": "", "Why it needs a human": v["reason"],
                             "Evidence": v["quote"], "Source": source})
    if not pool:
        st.info("Nothing here yet. Run an extraction in the Extract tab.")
    else:
        if any(r["Source"] == "scripted simulation" for r in rows) or any(s["mode"] == "sim" for s in st.session_state.runs.values()):
            st.warning(SIM_WARNING)
        a, b, c = st.columns(3)
        a.metric("Contracts processed", len(pool))
        b.metric("Fields needing a human", f"{len(rows)} of {total_fields}",
                 help=f"{100 * len(rows) / max(1, total_fields):.1f}% of all extracted fields")
        if rows:
            df = pd.DataFrame(rows)
            edited = st.data_editor(
                df, hide_index=True, width="stretch", key=f"review_{len(df)}",
                disabled=[c for c in df.columns if c not in ("Decision", "Human value")],
                column_config={"Contract": st.column_config.TextColumn(width=105),
                               "Field": st.column_config.TextColumn(width=150),
                               "Agent value": st.column_config.TextColumn(width=110),
                               "Decision": st.column_config.SelectboxColumn(
                                   options=["Pending", "Accept agent value", "Override"], required=True, width=165),
                               "Human value": st.column_config.TextColumn(width=120),
                               "Why it needs a human": st.column_config.TextColumn(width=270),
                               "Evidence": st.column_config.TextColumn(width=260),
                               "Source": st.column_config.TextColumn(width=130)})
            c.metric("Resolved in this session", int((edited["Decision"] != "Pending").sum()))
            st.caption("Decisions are not saved - this shows the workflow. In production they would write back to the "
                       "contract database and become labelled examples for improving the agent.")
        else:
            c.metric("Resolved in this session", 0)
            st.success("No fields needed review in the loaded runs.")


# ---------------------------------------------------------------- Accuracy tab
with tab_acc:
    st.subheader("Does the agent beat the rules approach?")
    st.markdown("Both are scored against the answer key on the same 30 synthetic contracts in 5 layouts. "
                "The **rules baseline** stands in for AArete's earlier positional-rules phase: regexes written for "
                "layouts A and B, first match wins, no notion of amendments or look-alike numbers.")
    rules = rules_results()
    agent_preds = {cid: {f: v["value"] for f, v in r["fields"].items()} for cid, r in CACHE.items()}
    agent = score(agent_preds, KEY) if agent_preds else None

    k = st.columns(3)
    k[0].metric("Rules baseline", f"{rules['overall']}%", help=f"Scored on {rules['n_contracts']} contracts")
    if agent:
        k[1].metric("Agent", f"{agent['overall']}%", f"{agent['overall'] - rules['overall']:+.1f} pts vs rules")
        k[2].metric("Agent contracts scored", agent["n_contracts"])
        if agent["n_contracts"] < len(KEY):
            st.info(f"Only {agent['n_contracts']} of {len(KEY)} contracts have saved agent runs, so the comparison "
                    "covers those only. Run `python run_agent.py --all` for the full set.")
    else:
        k[1].metric("Agent", "not run yet")
        st.info("No saved agent runs yet, so only the rules baseline is shown. Run `python run_agent.py --all` "
                "(needs a free Gemini key), commit `data/cache/`, and this page will compare both.")

    st.caption("Scoring is strict: a field sent to a human counts as unanswered, i.e. wrong. After human review the "
               "end-to-end result is better; the routing numbers below show how safe the auto-accepted fields are.")
    long = [{"Layout": LAYOUT_LABELS[l], "Method": "Rules baseline", "Accuracy": v} for l, v in rules["by_layout"].items()]
    if agent:
        long += [{"Layout": LAYOUT_LABELS[l], "Method": "Agent", "Accuracy": v} for l, v in agent["by_layout"].items()]
    long_df = pd.DataFrame(long)
    st.altair_chart(accuracy_chart(long_df, MODE), theme=None, width="stretch")
    with st.expander("View as table"):
        st.dataframe(long_df.pivot(index="Layout", columns="Method", values="Accuracy"), width="stretch")

    st.markdown("**By field**")
    by_field = pd.DataFrame({"Field": [FIELD_LABELS[f] for f in rules["by_field"]],
                             "Rules baseline (%)": list(rules["by_field"].values())})
    if agent:
        by_field["Agent (%)"] = [agent["by_field"][f] for f in rules["by_field"]]
    st.dataframe(by_field, hide_index=True, width="stretch")

    if agent:
        r = agent_routing(DATA, KEY)
        if r:
            st.markdown("**Is the agent's own confidence trustworthy?**")
            q = st.columns(3)
            q[0].metric("Fields auto-accepted", f"{r['auto_accepted_pct']}%")
            q[1].metric("Accuracy of auto-accepted fields", f"{r['auto_accuracy_pct']}%")
            q[2].metric("Sent to a human", f"{r['sent_to_review_pct']}%")
            extra = (f" {r['review_fields_that_were_actually_correct']} flagged fields were actually correct "
                     "(cautious, not wrong)." if r["review_fields_that_were_actually_correct"] else "")
            st.caption("High accuracy on auto-accepted fields is what makes the human-review savings safe." + extra)
    st.caption("Limits: synthetic contracts are cleaner than real ones, and the rules baseline's score depends on how many "
               "layouts its rules cover. Treat these numbers as a controlled comparison, not a forecast.")


# ---------------------------------------------------------------- Impact tab
with tab_impact:
    st.subheader("Business impact")
    st.markdown("Savings come from analyst time no longer spent reading contracts. Inputs from the AArete case study are "
                "marked; everything else is an **assumption you can change**.")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        vol = st.number_input("Contracts processed per year", 100, 1_000_000, 10_000, 500, key="vol")
        rate = st.number_input("Analyst cost per hour ($, fully loaded) - assumption", 5.0, 300.0, 40.0, 1.0, key="rate")
        docs_pw = st.number_input("Contracts one analyst handles per week - case study", 10, 1000, 100, 10, key="docs_pw")
        hrs = st.number_input("Working hours per week", 10.0, 80.0, 40.0, 1.0, key="hrs")
        llm_c = st.number_input("LLM/API cost per contract ($) - assumption", 0.0, 20.0, 0.30, 0.05, key="llm_c")
        plat = st.number_input("Platform cost per year ($: hosting, upkeep) - assumption, 0 = excluded", 0, 1_000_000, 0, 1000, key="plat")

        measured_share = review_share_from_results(list(CACHE.values())) if CACHE else None
        opts = ["Case study: 97% of effort saved (3% remains)", "Custom"]
        if measured_share is not None:
            opts.insert(1, f"Measured from {len(CACHE)} saved agent runs ({100 * measured_share:.1f}% of fields reviewed)")
        basis = st.radio("Share of effort that remains (human review)", opts, key="basis")
        silent, err_cost = 0.0, 0.0
        if basis.startswith("Case study"):
            share = 0.03
        elif basis == "Custom":
            share = st.slider("Share of manual effort remaining", 0.0, 1.0, 0.05, 0.01, key="share")
        else:
            share = measured_share
            silent = silent_errors_per_contract(CACHE, KEY) or 0.0
            err_cost = st.number_input(f"Cost of one wrong auto-accepted field ($) - assumption "
                                       f"(measured: {silent:.2f} per contract)", 0.0, 1000.0, 10.0, 1.0, key="err_cost")
        rec_on = st.toggle("Include overpayment-recovery benefit (illustrative)", value=False, key="rec_on")
        claims = leak = recr = 0.0
        if rec_on:
            claims = st.number_input("Annual claims paid under these contracts ($M)", 1.0, 100_000.0, 100.0, 10.0, key="claims")
            leak = st.number_input("% of that spend paid wrongly because terms weren't machine-readable", 0.0, 20.0, 0.5, 0.1, key="leak")
            recr = st.number_input("% of that leakage recovered once terms are structured", 0.0, 100.0, 50.0, 5.0, key="recr")

    inp = ImpactInputs(contracts_per_year=int(vol), docs_per_person_week=float(docs_pw), hours_per_week=float(hrs),
                       analyst_rate_usd_hr=float(rate), llm_cost_per_contract=float(llm_c), review_share=float(share),
                       platform_cost_per_year=float(plat), silent_errors_per_contract=float(silent),
                       cost_per_silent_error=float(err_cost), recovery_enabled=rec_on, claims_paid_musd=float(claims or 100),
                       misread_leakage_pct=float(leak), recovery_rate_pct=float(recr))
    r = compute(inp)
    with right:
        a, b = st.columns(2)
        a.metric("Manual cost / year", f"${r['manual_cost']:,.0f}")
        b.metric("With agent / year", f"${r['ai_cost']:,.0f}")
        c, d = st.columns(2)
        c.metric("Net savings / year", f"${r['savings']:,.0f}", f"-{r['savings_pct']:.0f}% vs manual cost",
                 delta_color="inverse")
        d.metric(f"Analyst FTE freed (of {r['fte_manual']:.1f})", f"{r['fte_manual'] - r['fte_after']:.1f}")
        st.altair_chart(cost_chart(r["manual_cost"], r["ai_cost"], MODE), theme=None, width="stretch")
        st.markdown(f"Per contract: **\\${r['cost_per_contract_manual']:.2f}** manual → **\\${r['cost_per_contract_ai']:.2f}** "
                    f"with the agent ({r['minutes_per_contract']:.0f} analyst-minutes per contract today).")
        st.dataframe(pd.DataFrame({
            "Cost line": ["LLM / API", "Human review of flagged fields", "Wrong auto-accepted fields", "Platform", "Total with agent"],
            "$ / year": [r["llm_cost"], r["review_cost"], r["error_cost"], r["platform_cost"], r["ai_cost"]]}),
            hide_index=True, width="stretch", column_config={"$ / year": st.column_config.NumberColumn(format="$%,.0f")})
        if rec_on:
            st.success(f"Illustrative recovery benefit: \\${r['recovery']:,.0f}/yr → total benefit \\${r['total_benefit']:,.0f}/yr. "
                       "Only claim this if you can defend the leakage and recovery rates.")
    st.markdown("**Sensitivity: net annual savings**")
    st.dataframe(sensitivity(inp).style.format("${:,.0f}"), width="stretch")
    st.caption("From the case study: ~100 contracts per person per week manually and a 97% reduction in manual effort. "
               "Volume, analyst cost, API cost, platform cost, error cost and recovery are assumptions. The case study's "
               "$330M client-savings figure is not reproduced here because its cost base is not disclosed.")


# ---------------------------------------------------------------- How it works tab
with tab_how:
    st.subheader("How the agent works")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6", fontname="Helvetica", fontsize=11];
      edge [color="#52514e", fontname="Helvetica", fontsize=10];
      pdf [label="Contract PDF"];
      llm [label="LLM agent\\nplans, picks tools"];
      tools [label="Tools\\nsearch · read chunk\\nsubmit · flag · finish"];
      ver [label="Verifier (plain code)\\nquote exists in contract?\\nvalue appears in quote?\\nplausible? dates consistent?", fillcolor="#fdf0ea", color="#eb6834"];
      ok [label="Auto-accepted fields"];
      hum [label="Human review queue"];
      pdf -> llm; llm -> tools; tools -> ver; ver -> llm [label="issues → fix and retry"];
      ver -> ok [label="verified + confident"]; llm -> hum [label="flag / unsure"];
    }""")
    st.markdown("""
**Why this is an agent, not a script.** The LLM decides what to read, when it has enough evidence, and when to give up
and flag a field. Its answers are not trusted: a separate verifier checks every quote against the contract text, and
failures are returned to the agent as tool results so it must re-read and correct itself.

**Domain rules in the prompt** (the equivalent of AArete's "17 years of ontology"): amendments override base terms;
ignore superseded terms; don't confuse emergency or behavioral-health rates with the acute/general rates; separate
the appeals window from the filing window; read numbers written as words.

**What is real and what is not.**
- Contracts are synthetic (fictional parties), in five layouts, with look-alike numbers and an answer key.
- *Replay* and *Live* modes show genuine Gemini output; *Simulation* is a script for testing the interface.
- Nothing is saved from the review queue, and scanned PDFs are not supported.
""")
    st.markdown(f"Source case study: [{CASE_URL.split('/')[-1][:60]}…]({CASE_URL})")
