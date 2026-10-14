"""A SCRIPTED stand-in for the LLM, used ONLY to test the agent's plumbing offline.

It knows the answer key, so its accuracy says nothing about Gemini. What it does prove:
the tool loop, quote verification, issue feedback, multi-call turns, flagging, finish and
routing all work. It deliberately makes two mistakes on its first submission (a fabricated
quote and a wrong value) so the self-correction path is exercised on every contract.
"""
from .doc import ContractDoc
from .llm import ModelTurn, ToolCall
from .tools import FIELD_NAMES, check_field, coerce


def support_quote(doc: ContractDoc, field: str, value):
    """Find 1-3 consecutive lines of the contract that pass the real validator for (field, value)."""
    for c in doc.chunks:
        lines = [ln for ln in c["text"].splitlines() if ln.strip()]
        for size in (1, 2, 3):
            for i in range(len(lines) - size + 1):
                q = " ".join(lines[i:i + size])
                if not check_field(doc, field, value, q):
                    return q
    return None


class MockLLM:
    def __init__(self, doc: ContractDoc, truth: dict, inject_errors=True, flag_field=None):
        self.doc, self.truth = doc, truth
        self.inject, self.flag_field = inject_errors, flag_field
        self.bad_fabricated, self.bad_value = "timely_filing_days", "inpatient_per_diem_usd"
        if flag_field in (self.bad_fabricated, self.bad_value):
            raise ValueError("pick a flag_field that is not one of the injected-error fields")

    def _item(self, field, value=None, quote=None):
        v = self.truth[field] if value is None else value
        q = quote if quote is not None else support_quote(self.doc, field, coerce(field, v)[0])
        return {"field": field, "value": str(v).lower() if isinstance(v, bool) else str(v),
                "quote": q or "", "confidence": 0.92}

    def step(self, system, messages, tools) -> ModelTurn:
        turn = sum(1 for m in messages if m["role"] == "model")
        usage = {"prompt": 1000, "output": 120}
        if turn == 0:
            calls = [ToolCall("search_contract", {"query": "effective term expire renew"})]
        elif turn == 1:
            items = []
            for f in FIELD_NAMES:
                if f == self.flag_field:
                    continue
                if self.inject and f == self.bad_fabricated:
                    items.append(self._item(f, quote="The provider must submit all claims promptly."))
                elif self.inject and f == self.bad_value:
                    items.append(self._item(f, value=self.truth[f] + 100,
                                            quote=support_quote(self.doc, f, self.truth[f])))
                else:
                    items.append(self._item(f))
            calls = [ToolCall("submit_fields", {"fields": items})]
        elif turn == 2:
            calls = []
            if self.inject:
                calls.append(ToolCall("submit_fields", {"fields": [self._item(self.bad_fabricated),
                                                                  self._item(self.bad_value)]}))
            if self.flag_field:
                calls.append(ToolCall("flag_for_human", {"field": self.flag_field,
                                                         "reason": "mock: contract unclear"}))
            if not calls:
                calls = [ToolCall("finish", {})]
        else:
            calls = [ToolCall("finish", {})]
        return ModelTurn(text=f"mock turn {turn}", tool_calls=calls, raw={"mock": turn}, usage=usage)
