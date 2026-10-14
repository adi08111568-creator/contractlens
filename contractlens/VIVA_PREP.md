# ContractLens - Viva preparation

Source case: AArete "Doxy AI", ZenML LLMOps Database (healthcare contract metadata extraction).

## 1. The business problem (30 seconds)
Payers and providers sign contracts with hundreds of pages of terms: rates, filing deadlines, payment
timelines, renewal and termination rules. Analysts read each contract by hand (~100 per person per week in
the case study) and key terms into systems. It is slow and error-prone, and mistakes mean wrongly paid claims.

## 2. The solution
An **AI agent** that reads a contract, finds each term, and **proves every value with a verbatim quote**.
Code checks the proof. Wrong or unprovable values go back to the agent to fix. Only fields it still cannot
prove go to a human. So humans review exceptions, not every document.

## 3. Why an agent, not just an LLM prompt
- It decides what to read next (search, read a section), uses tools, and acts on feedback.
- The environment verifies its work (quote exists, value is in the quote, plausible range, expiry after effective).
- It knows when to stop and ask a human (`flag_for_human`).
A single prompt cannot check itself or loop on its errors.

## 4. Architecture
Contract PDF -> text -> outline/chunks -> Agent (Gemini, tool calling) with tools
`search_contract`, `read_chunk`, `submit_fields`, `flag_for_human`, `finish`
-> Verifier (code) -> each field `auto` or `review` -> Streamlit UI (Extract, Review queue, Accuracy, Impact, How it works).

## 5. Success criteria
- Field accuracy vs answer key, and vs a rules baseline.
- Share of fields auto-accepted vs sent to humans.
- **Silent errors**: wrong values that were auto-accepted (the costly kind).
- Annual cost vs manual.

## 6. Impact math (change inputs live in the Impact tab)
Grounded in the case study: ~100 contracts/person/week; 97% manual effort reduction.
Assumptions (mine): 10,000 contracts/yr, $40/hr loaded analyst, $0.30 LLM cost/contract.
- Manual = 10,000 x 24 min / 60 x $40 = **$160,000**
- With agent = LLM $3,000 + human review 3% of manual $4,800 = **$7,800**
- Net savings = **$152,200 (95%)**; ~2 analyst FTE freed.
Optional, off by default and clearly illustrative: overpayment recovery from structured terms.
Include the sensitivity grid (volume x hourly rate) and the platform and silent-error cost lines.
Say plainly: the 97% is the case study's claim; for my prototype, use the **measured** review share (Impact tab, "Measured" option).

## 7. Results (fill in after your real run)
Rules baseline (measured, 30 synthetic contracts): **59.2%** overall - 100% on layouts A and B (what the rules were written for), 16.7% on C, 54.2% on D, 25% on E.
Agent: ____% overall, ____% auto-accepted, ____ silent errors. (from `python evaluate.py --method both`)
Message: rules break when layouts change; the agent generalises, and verification makes errors visible.

## 8. Limitations - say these before they ask
- Contracts are **synthetic** (5 layouts, fictional parties). Real contracts are messier; scanned PDFs need OCR (the case study used Textract).
- Only 12 fields; real systems have hundreds.
- 30 contracts is small; accuracy numbers are indicative, not production-grade.
- Impact figures other than the case-study numbers are assumptions.
- Quote verification proves a value is *in the text*, not that it is the *right* clause. Hence confidence thresholds and human review.
- Review decisions in the UI are not stored (workflow demo only).

## 9. Likely questions
**Why Gemini?** Free tier; the LLM client is swappable (case study used Claude on AWS Bedrock).
**How do you stop hallucination?** Quote must exist verbatim in the document; value must appear in the quote; range and cross-field checks; failures return to the agent.
**What is the human's role?** Exceptions only, and their corrections become labelled data.
**What if the amendment changes a rate?** The prompt tells the agent amendments override; layout D tests this.
**How would you scale to 500K docs/week?** Queue + parallel workers, batch API, OCR pipeline, cache, monitoring of silent-error rate (the case study used AWS and Snowflake).
**Is 95% savings realistic?** It is the case study's headline direction with my assumptions; the sensitivity grid and measured review share show how it moves.
**What would you do next?** Real contracts, more fields, evaluation on a held-out set, feedback loop from reviewer corrections, cost and latency monitoring.
**Biggest risk?** Silent errors: confident but wrong. Measured and priced in the model.
