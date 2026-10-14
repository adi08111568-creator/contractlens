"""Offline tests for the agent's plumbing. Run:  python tests/test_plumbing.py   (or pytest).

These do NOT measure LLM accuracy - Gemini is not called. They check that the tools, the
verification loop, routing, the Gemini adapter and the scorer behave as designed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent import run_agent                      # noqa: E402
from agent.doc import ContractDoc, load_contract, norm  # noqa: E402
from agent.llm import GeminiClient, ModelTurn, ToolCall  # noqa: E402
from agent.mock_llm import MockLLM                      # noqa: E402
from agent.tools import FIELD_NAMES, Session            # noqa: E402
from evaluate import score                              # noqa: E402

DATA = ROOT / "data"
KEY = json.loads((DATA / "ground_truth.json").read_text())
PDFS = sorted((DATA / "contracts" / "pdf").glob("*.pdf"))


def test_chunks_cover_all_text():
    for p in PDFS:
        d = load_contract(p)
        assert norm("".join(c["text"] for c in d.chunks)) == norm(d.text), p.stem
        assert all(len(c["text"]) < 2200 for c in d.chunks), p.stem


def test_validator_catches_bad_submissions():
    d = load_contract(DATA / "contracts" / "pdf" / "contract_18.pdf")   # layout D (has an amendment)
    s = Session(d)
    r = s.submit_fields([
        {"field": "inpatient_per_diem_usd", "value": "3400",
         "quote": "Acute inpatient services shall be reimbursed at a per diem of $3,400."},   # good (amended)
        {"field": "outpatient_pct_of_medicare", "value": "120",
         "quote": "General outpatient services shall be reimbursed at 115% of the Medicare Fee Schedule."},  # wrong value
        {"field": "clean_claim_payment_days", "value": "45",
         "quote": "The Plan pays every claim in 45 days."},                                   # paraphrase
        {"field": "timely_filing_days", "value": "30", "quote": "x"},                         # quote too short
        {"field": "credentialing_sla_days", "value": "9000",
         "quote": "The Plan shall complete credentialing within 90 days"},                    # value not in quote
        {"field": "governing_law", "value": "Texas",
         "quote": "This Agreement is governed by the laws of the State of North Carolina."},  # wrong state
    ])
    assert r["verified"] == ["inpatient_per_diem_usd"], r
    bad = {i["field"] for i in r["issues"]}
    assert bad == {"outpatient_pct_of_medicature".replace("medicature", "medicare"), "clean_claim_payment_days",
                   "timely_filing_days", "credentialing_sla_days", "governing_law"}, bad


def test_quote_matching_tolerates_formatting_but_not_paraphrase():
    d = ContractDoc("The Plan shall pay each clean claim within thirty-\n five (35) days after receipt.")
    assert d.contains_quote("pay each clean claim within thirty-five (35) days")
    assert not d.contains_quote("pay every clean claim within 35 days")


def test_cross_field_dates_and_finish_guard():
    d = load_contract(DATA / "contracts" / "pdf" / "contract_18.pdf")
    s0 = Session(d)
    assert "error" in s0.finish() and not s0.finished
    doc = ContractDoc("This Agreement begins on 10 Oct 2025.\nThe initial term expires on 11 Oct 2024.")
    s = Session(doc)
    r = s.submit_fields([
        {"field": "effective_date", "value": "2025-10-10", "quote": "This Agreement begins on 10 Oct 2025."},
        {"field": "expiry_date", "value": "2024-10-11", "quote": "The initial term expires on 11 Oct 2024."},
    ])
    # each passes its own checks; only the cross-field rule (expiry after effective) can catch this
    assert any("after the effective" in i["problem"] for i in r["issues"]), r
    assert s.fields["expiry_date"]["ok"] is False and s.fields["effective_date"]["ok"] is False


def test_anchor_does_not_match_inside_other_words():
    d = load_contract(DATA / "contracts" / "pdf" / "contract_18.pdf")
    s = Session(d)
    r = s.submit_fields([{"field": "expiry_date", "value": "2024-10-11",
                          "quote": "Effective October 11, 2024, the parties amend Sections 3.1 and 3.2"}])
    assert any("does not seem to address" in i["problem"] for i in r["issues"]), r   # 'amend' is not 'end'


def test_mock_agent_all_contracts_self_corrects_and_scores_perfectly():
    preds = {}
    for p in PDFS:
        truth = KEY[p.stem]["truth"]
        doc = load_contract(p)
        res = run_agent(doc, MockLLM(doc, truth))
        assert res["stop_reason"] == "finished", (p.stem, res["stop_reason"])
        assert res["validation_issues_caught"] >= 2, p.stem          # injected errors were caught
        assert res["auto_fields"] == 12 and res["review_fields"] == 0, p.stem
        preds[p.stem] = {f: v["value"] for f, v in res["fields"].items()}
    s = score(preds, KEY)
    assert s["overall"] == 100.0, s["by_field"]                      # plumbing is lossless


def test_flagging_routes_to_review():
    p = PDFS[0]
    doc = load_contract(p)
    res = run_agent(doc, MockLLM(doc, KEY[p.stem]["truth"], flag_field="governing_law"))
    assert res["stop_reason"] == "finished"
    assert res["fields"]["governing_law"]["status"] == "review"
    assert "flagged by agent" in res["fields"]["governing_law"]["reason"]
    assert res["auto_fields"] == 11


def test_low_confidence_routes_to_review():
    d = load_contract(DATA / "contracts" / "pdf" / "contract_18.pdf")
    s = Session(d)
    s.submit_fields([{"field": "governing_law", "value": "North Carolina", "confidence": 0.5,
                      "quote": "This Agreement is governed by the laws of the State of North Carolina."}])
    f = s.result(threshold=0.75)["governing_law"]
    assert f["status"] == "review" and f["value"] == "North Carolina" and "low confidence" in f["reason"]


class _Scripted:
    """Tiny fake LLM driven by a list of turns."""
    def __init__(self, turns):
        self.turns, self.i = turns, 0

    def step(self, system, messages, tools):
        t = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return t


def test_loop_nudges_a_model_that_stops_early_and_respects_max_steps():
    d = load_contract(PDFS[0])
    chatty = _Scripted([ModelTurn(text="Done!", raw={"x": 1})])           # never calls a tool
    res = run_agent(d, chatty, max_steps=6)
    assert res["stop_reason"] == "no_tool_call" and res["steps"] == 3     # 1 + 2 nudges, then gives up
    assert res["review_fields"] == 12

    looper = _Scripted([ModelTurn(tool_calls=[ToolCall("search_contract", {"query": "term"})], raw={"x": 1})])
    res = run_agent(d, looper, max_steps=4)
    assert res["stop_reason"] == "max_steps" and res["steps"] == 4 and res["review_fields"] == 12


def test_unknown_tool_and_bad_args_do_not_crash():
    s = Session(load_contract(PDFS[0]))
    assert "error" in s.call("nope", {})
    assert "error" in s.call("read_chunk", {"wrong": 1})
    assert "error" in s.call("read_chunk", {"chunk_id": 999})


# ---------------------------------------------------------------- Gemini adapter (no network)
class _FakeModels:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _FakeClient:
    def __init__(self, responses):
        self.models = _FakeModels(responses)


def _resp(parts, usage=(10, 5)):
    from google.genai import types
    return types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": parts}}],
        "usage_metadata": {"prompt_token_count": usage[0], "candidates_token_count": usage[1]}})


def test_gemini_adapter_parses_calls_and_round_trips_messages():
    fake = _FakeClient([_resp([{"text": "reading"}, {"function_call": {"name": "read_chunk", "args": {"chunk_id": 2}}}])])
    g = GeminiClient(_client=fake, model="test-model")
    turn = g.step("sys", [{"role": "user", "text": "hi"}], [
        {"name": "read_chunk", "description": "d", "parameters": {"type": "object", "properties": {}}}])
    assert turn.text == "reading" and turn.tool_calls == [ToolCall("read_chunk", {"chunk_id": 2})]
    assert turn.usage == {"prompt": 10, "output": 5}
    sent = fake.models.calls[0]
    assert sent["model"] == "test-model" and sent["config"].temperature == 0.0
    assert sent["config"].automatic_function_calling.disable is True
    # next request: model turn goes back as its raw object, tool results as function_response parts
    msgs = [{"role": "user", "text": "hi"}, {"role": "model", "text": "", "tool_calls": turn.tool_calls, "raw": turn.raw},
            {"role": "tool", "results": [{"name": "read_chunk", "response": {"text": "abc"}}]}]
    contents = g._contents(msgs)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[2].parts[0].function_response.name == "read_chunk"


def test_gemini_adapter_retries_rate_limits_then_gives_up_on_other_errors():
    import time as _t
    slept = []
    real_sleep, _t.sleep = _t.sleep, lambda s: slept.append(s)
    try:
        ok = _resp([{"text": "ok"}])
        g = GeminiClient(_client=_FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED retry in 2.5s"), ok]), model="m")
        assert g.step("s", [{"role": "user", "text": "x"}], []).text == "ok" and len(slept) == 1
        g2 = GeminiClient(_client=_FakeClient([ValueError("400 invalid argument")]), model="m")
        try:
            g2.step("s", [{"role": "user", "text": "x"}], [])
            raise AssertionError("should have raised")
        except ValueError:
            pass
    finally:
        _t.sleep = real_sleep


def test_gemini_adapter_handles_empty_response():
    from google.genai import types
    empty = types.GenerateContentResponse.model_validate({"candidates": []})
    g = GeminiClient(_client=_FakeClient([empty]), model="m")
    t = g.step("s", [{"role": "user", "text": "x"}], [])
    assert t.tool_calls == [] and "empty response" in t.text


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
