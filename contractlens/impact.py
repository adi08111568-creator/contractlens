"""Business-impact model for ContractLens. Pure functions - no Streamlit, easy to test.

Grounded inputs (from the AArete case study in the ZenML LLMOps Database):
  * ~100 contracts per person per week when processed manually
  * 97% reduction in manual effort with Doxy AI
Everything else (volume, analyst cost, LLM cost, platform cost, recovery) is an ASSUMPTION
that the user can change in the app.
"""
from dataclasses import dataclass, asdict

import pandas as pd

WORK_WEEKS_PER_YEAR = 48


@dataclass
class ImpactInputs:
    contracts_per_year: int = 10_000
    docs_per_person_week: float = 100        # case study: manual rate per person
    hours_per_week: float = 40
    analyst_rate_usd_hr: float = 40          # assumption: fully loaded
    llm_cost_per_contract: float = 0.30      # assumption: production API cost per contract
    review_share: float = 0.03               # share of manual effort that remains (case study: 97% saved)
    platform_cost_per_year: float = 0.0      # assumption: hosting + maintenance (0 = excluded)
    silent_errors_per_contract: float = 0.0  # measured: auto-accepted but wrong fields per contract
    cost_per_silent_error: float = 0.0       # assumption: cost of one undetected wrong field
    recovery_enabled: bool = False           # optional, illustrative lever
    claims_paid_musd: float = 100.0          # annual claims paid under these contracts, $M
    misread_leakage_pct: float = 0.5         # % of that spend paid wrongly because terms were not machine-readable
    recovery_rate_pct: float = 50.0          # % of the leakage that is recovered once terms are structured


def compute(i: ImpactInputs) -> dict:
    minutes = i.hours_per_week * 60 / i.docs_per_person_week
    manual_hours = i.contracts_per_year * minutes / 60
    manual_cost = manual_hours * i.analyst_rate_usd_hr
    llm_cost = i.contracts_per_year * i.llm_cost_per_contract
    review_cost = manual_cost * i.review_share
    error_cost = i.contracts_per_year * i.silent_errors_per_contract * i.cost_per_silent_error
    ai_cost = llm_cost + review_cost + error_cost + i.platform_cost_per_year
    savings = manual_cost - ai_cost
    recovery = (i.claims_paid_musd * 1e6 * i.misread_leakage_pct / 100 * i.recovery_rate_pct / 100
                if i.recovery_enabled else 0.0)
    fte_manual = i.contracts_per_year / (i.docs_per_person_week * WORK_WEEKS_PER_YEAR)
    return {
        "minutes_per_contract": minutes,
        "manual_hours": manual_hours,
        "manual_cost": manual_cost,
        "llm_cost": llm_cost,
        "review_cost": review_cost,
        "error_cost": error_cost,
        "platform_cost": i.platform_cost_per_year,
        "ai_cost": ai_cost,
        "savings": savings,
        "savings_pct": 100 * savings / manual_cost if manual_cost else 0.0,
        "recovery": recovery,
        "total_benefit": savings + recovery,
        "fte_manual": fte_manual,
        "fte_after": fte_manual * i.review_share,
        "cost_per_contract_manual": manual_cost / i.contracts_per_year if i.contracts_per_year else 0.0,
        "cost_per_contract_ai": ai_cost / i.contracts_per_year if i.contracts_per_year else 0.0,
    }


def sensitivity(i: ImpactInputs, volumes=(1_000, 5_000, 10_000, 50_000), rates=(20, 40, 60)) -> pd.DataFrame:
    """Net annual savings (USD) for volume x analyst-rate combinations, other inputs held fixed."""
    rows = {}
    for r in rates:
        rows[f"${r}/hr"] = [compute(ImpactInputs(**{**asdict(i), "contracts_per_year": v,
                                                    "analyst_rate_usd_hr": r}))["savings"] for v in volumes]
    return pd.DataFrame(rows, index=[f"{v:,} contracts/yr" for v in volumes])


def review_share_from_results(results) -> float | None:
    """Average share of fields sent to human review, from agent runs (a proxy for residual effort)."""
    shares = [r["review_fields"] / max(1, r["auto_fields"] + r["review_fields"]) for r in results]
    return sum(shares) / len(shares) if shares else None
