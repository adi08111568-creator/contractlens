"""The extraction agent: a tool-calling loop around an LLM, with the environment verifying every answer."""
from .doc import ContractDoc
from .tools import FIELD_SPECS, TOOL_SPECS, Session

SYSTEM_PROMPT = """You are ContractLens, a contract-analysis agent for a healthcare payer's contracting team.
Your job: extract 12 key terms from ONE provider contract, with proof.

You work through tools, not memory:
- search_contract(query) / read_chunk(chunk_id): read the contract. You start with an outline only.
- submit_fields(fields): propose values. Every value needs a VERBATIM quote from the contract. The tool verifies
  quotes and values and returns issues. If it reports issues, re-read the relevant clause and fix them.
- flag_for_human(field, reason): use when the contract is ambiguous, contradictory or silent. Never guess.
- finish(): only when every field is verified or flagged.

Plan: skim the outline, read/search the clauses you need, submit all fields in one call, fix any issues, finish.
Be efficient: batch your submissions, do not re-read chunks you have already seen.

Domain rules (these decide most mistakes):
1. Contracts contain look-alike numbers. Extract exactly what each field means:
   - Inpatient per diem = ACUTE medical/surgical only. Ignore behavioral-health and emergency rates.
   - Outpatient % = GENERAL outpatient only. Ignore the emergency department percentage.
   - Termination notice = termination WITHOUT cause. Ignore non-renewal notice and dispute periods.
   - Timely filing = days for the PROVIDER to submit claims. Ignore the appeals window.
   - Clean-claim payment = days for the PLAN to pay. Credentialing = days for the PLAN to credential.
2. Amendments override the base agreement. If an amendment replaces a rate, report the AMENDED rate.
3. Superseded or prior-agreement terms ("replaces the earlier arrangement ... no longer apply") are history. Ignore them.
4. Numbers may be written in words, e.g. "sixty (60)" or "one hundred five (105) percent". Report digits only.
5. Dates may appear in different formats; always output YYYY-MM-DD.
6. auto_renewal is true only if the contract renews automatically; "requires a signed amendment to extend" means false.
7. Abbreviated term sheets: MCR FS = Medicare fee schedule, CCP = clean claim payment, DOS = date of service,
   IP = inpatient, OP = outpatient.
8. Quotes: copy text exactly. For tables, quote the whole ROW including its label (e.g. 'Expiration Date 24 Sep 2024', not just '24 Sep 2024').
9. Confidence: 0.9+ when one clause states the value plainly; 0.6-0.8 if you had to interpret; flag if unsure.

Fields to extract:
""" + "\n".join(f"- {n} ({s['kind']}): {s['desc']}" for n, s in FIELD_SPECS.items())


def run_agent(doc: ContractDoc, client, max_steps: int = 12, threshold: float = 0.75, on_step=None):
    """Run the agent on one contract. Returns dict with fields, trace and run stats.

    on_step(entry) is called after every step so a UI can stream the agent's work live.
    """
    session = Session(doc)
    messages = [{"role": "user", "text":
                 f"Contract: {doc.name}\n\nOutline (chunk id, first words):\n{doc.outline()}\n\n"
                 "Extract all 12 fields."}]
    trace, usage = [], {"prompt": 0, "output": 0}
    stop, nudges = "max_steps", 0

    for step in range(1, max_steps + 1):
        turn = client.step(SYSTEM_PROMPT, messages, TOOL_SPECS)
        for k in usage:
            usage[k] += turn.usage.get(k, 0)
        entry = {"step": step, "thought": turn.text, "calls": []}

        if not turn.tool_calls:
            trace.append(entry)
            if on_step:
                on_step(entry)
            if session.unresolved() and nudges < 2 and turn.raw is not None:
                nudges += 1
                messages.append({"role": "model", "text": turn.text, "tool_calls": [], "raw": turn.raw})
                messages.append({"role": "user", "text":
                                 f"You still have unresolved fields: {session.unresolved()}. "
                                 "Use the tools to extract and verify them, or flag them for a human."})
                continue
            stop = "no_tool_call"
            break

        results = []
        for tc in turn.tool_calls:
            out = session.call(tc.name, tc.args)
            entry["calls"].append({"tool": tc.name, "args": tc.args, "result": out})
            results.append({"name": tc.name, "response": out})
        trace.append(entry)
        if on_step:
            on_step(entry)
        messages.append({"role": "model", "text": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw})
        messages.append({"role": "tool", "results": results})
        if session.finished:
            stop = "finished"
            break

    fields = session.result(threshold)
    n_issues = sum(len(c["result"].get("issues", [])) for e in trace for c in e["calls"]
                   if c["tool"] == "submit_fields" and isinstance(c["result"], dict))
    return {
        "contract": doc.name,
        "fields": fields,
        "stop_reason": stop,
        "steps": len(trace),
        "validation_issues_caught": n_issues,
        "auto_fields": sum(1 for f in fields.values() if f["status"] == "auto"),
        "review_fields": sum(1 for f in fields.values() if f["status"] == "review"),
        "tokens": usage,
        "trace": trace,
    }
