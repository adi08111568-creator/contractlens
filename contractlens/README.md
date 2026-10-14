# ContractLens - contract extraction agent + Streamlit app

Use case: AArete "Doxy AI" from the ZenML LLMOps Database - extract structured terms from
provider contracts, with human review only for uncertain fields.

## What is the agent?
An LLM that works through tools in a loop and is checked by its environment:

    outline -> search_contract / read_chunk -> submit_fields(value + verbatim quote)
            -> code verifies quote, value, range, cross-field logic -> issues returned
            -> agent re-reads and fixes -> finish() or flag_for_human()

Every field ends as `auto` (verified, confident) or `review` (flagged / unverified / low confidence).

## Run it
    pip install -r requirements.txt
    streamlit run app.py                   # the app (works without a key: replay / simulation)
    python tests/test_plumbing.py          # offline tests, no key needed (13 tests)
    python tests/test_impact.py            # 6 tests
    python tests/test_app.py               # 8 UI tests
    python evaluate.py --method rules      # rules baseline score (no key needed)

    export GEMINI_API_KEY=...              # free key: https://aistudio.google.com
    python run_agent.py contract_18        # try ONE contract first (a layout-D contract with an amendment)
    python run_agent.py --all              # all 30, results cached in data/cache/agent/
    python evaluate.py --method both       # agent vs rules, by field and by layout

Model: defaults to `gemini-2.5-flash`; override with `GEMINI_MODEL` or `--model` if Google has
renamed it or your free tier offers a newer one. Free-tier rate limits: the client backs off
and retries on 429s; raise `--sleep` if you see many.

## Docs
- `DEPLOYMENT.md` - Gemini key, saving real runs, Streamlit Cloud steps
- `VIVA_PREP.md` - business case, impact math, limitations, likely questions

## Files
- `generate_contracts.py` - synthetic contracts (5 layouts) + ground truth (`data/ground_truth.*`)
- `agent/doc.py` - loading (PDF via pypdf), chunking, search, quote matching
- `agent/tools.py` - the 5 tools, field definitions, validators, review routing
- `agent/llm.py` - Gemini adapter (manual function calling, retries); provider-swappable
- `agent/agent.py` - system prompt (domain rules) and the loop
- `agent/mock_llm.py` - scripted stand-in used ONLY by tests
- `baseline_rules.py` - regex baseline written for layouts A and B
- `app.py`, `ui_helpers.py`, `impact.py` - Streamlit app and impact model
- `evaluate.py`, `run_agent.py`, `tests/`

## Honest status
- The agent loop and tools are tested offline with a scripted stand-in; **Gemini has not been
  called yet**, so real-model accuracy is unmeasured until you run `run_agent.py`.
- The rules baseline score depends on how many layouts the rules cover (A and B here).
