"""LLM provider layer. The agent loop only knows about `ModelTurn`; swapping providers
means writing one more class with a `step()` method.

Message format used by the loop (provider-neutral):
    {"role": "user",  "text": "..."}
    {"role": "model", "text": "...", "tool_calls": [ToolCall, ...], "raw": <provider object>}
    {"role": "tool",  "results": [{"name": "...", "response": {...}}, ...]}
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"   # override with GEMINI_MODEL; check AI Studio for current free models


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    raw: Any = None
    usage: dict = field(default_factory=dict)


class GeminiClient:
    """Gemini via the google-genai SDK with manual function calling (we run the loop ourselves)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_retries: int = 5, _client: Any = None):
        from google.genai import types
        self.types = types
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.max_retries = max_retries
        if _client is not None:           # injected fake for offline tests
            self.client = _client
        else:
            from google import genai
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("No Gemini API key. Set GEMINI_API_KEY (free key: aistudio.google.com).")
            self.client = genai.Client(api_key=key)

    # ---- conversion
    def _contents(self, messages):
        t = self.types
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part.from_text(text=m["text"])]))
            elif m["role"] == "model":
                out.append(m["raw"])      # keep the provider's own object (thought signatures etc.)
            elif m["role"] == "tool":
                out.append(t.Content(role="user", parts=[
                    t.Part.from_function_response(name=r["name"], response=r["response"])
                    for r in m["results"]]))
        return out

    def _config(self, system, tools):
        t = self.types
        decls = [t.FunctionDeclaration(name=s["name"], description=s["description"],
                                       parameters_json_schema=s["parameters"]) for s in tools]
        return t.GenerateContentConfig(
            system_instruction=system, tools=[t.Tool(function_declarations=decls)], temperature=0.0,
            max_output_tokens=4096,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))

    # ---- one model turn, with retry on rate limits / transient errors
    def step(self, system, messages, tools) -> ModelTurn:
        contents, config = self._contents(messages), self._config(system, tools)
        delay = 6.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                return self._parse(resp)
            except Exception as e:  # noqa: BLE001 - SDK raises several error types
                msg = str(e)
                transient = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE",
                                                   "DEADLINE", "overloaded"))
                if not transient or attempt == self.max_retries:
                    raise
                m = re.search(r"retry in ([\d.]+)s", msg, re.I) or re.search(r"retryDelay\W+(\d+)", msg)
                time.sleep(min(float(m.group(1)) + 1 if m else delay, 70))
                delay = min(delay * 2, 60)

    def _parse(self, resp) -> ModelTurn:
        if not resp.candidates or resp.candidates[0].content is None:
            reason = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else "no candidates"
            return ModelTurn(text=f"[empty response: {reason}]", raw=None)
        content = resp.candidates[0].content
        text, calls = [], []
        for p in content.parts or []:
            if getattr(p, "function_call", None):
                calls.append(ToolCall(p.function_call.name, dict(p.function_call.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text.append(p.text)
        u = resp.usage_metadata
        usage = {"prompt": getattr(u, "prompt_token_count", 0) or 0,
                 "output": getattr(u, "candidates_token_count", 0) or 0} if u else {}
        return ModelTurn(text="\n".join(text).strip(), tool_calls=calls, raw=content, usage=usage)
