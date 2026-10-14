"""Tests for the impact model. Run:  python tests/test_impact.py  (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from impact import ImpactInputs, compute, review_share_from_results, sensitivity  # noqa: E402


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


def test_defaults_reproduce_the_step2_business_case():
    r = compute(ImpactInputs())
    assert approx(r["minutes_per_contract"], 24.0)                 # 40h*60 / 100 docs
    assert approx(r["manual_cost"], 160_000)                       # 10,000 * 24/60 * $40
    assert approx(r["llm_cost"], 3_000) and approx(r["review_cost"], 4_800)
    assert approx(r["ai_cost"], 7_800) and approx(r["savings"], 152_200)
    assert approx(r["savings_pct"], 95.125, 0.001)
    assert approx(r["fte_manual"], 10_000 / (100 * 48)) and approx(r["fte_after"], r["fte_manual"] * 0.03)


def test_platform_and_error_costs_reduce_savings():
    base = compute(ImpactInputs())["savings"]
    plat = compute(ImpactInputs(platform_cost_per_year=12_000))["savings"]
    err = compute(ImpactInputs(silent_errors_per_contract=0.2, cost_per_silent_error=10))["savings"]
    assert approx(base - plat, 12_000) and approx(base - err, 10_000 * 0.2 * 10)


def test_recovery_lever_is_off_by_default_and_additive_when_on():
    assert compute(ImpactInputs())["recovery"] == 0.0
    on = compute(ImpactInputs(recovery_enabled=True))
    assert approx(on["recovery"], 100e6 * 0.005 * 0.5)             # $250k illustrative
    assert approx(on["total_benefit"], on["savings"] + on["recovery"])


def test_more_review_means_less_saving_and_zero_volume_is_safe():
    assert compute(ImpactInputs(review_share=0.2))["savings"] < compute(ImpactInputs())["savings"]
    z = compute(ImpactInputs(contracts_per_year=0))
    assert z["manual_cost"] == 0 and z["savings_pct"] == 0 and z["cost_per_contract_ai"] == 0


def test_sensitivity_grid_shape_and_monotonic():
    s = sensitivity(ImpactInputs())
    assert s.shape == (4, 3)
    assert (s.diff().iloc[1:] > 0).all().all()                     # more volume -> more savings
    assert (s.diff(axis=1).iloc[:, 1:] > 0).all().all()            # higher analyst rate -> more savings
    assert approx(s.loc["10,000 contracts/yr", "$40/hr"], 152_200)  # matches the headline case


def test_review_share_from_results():
    runs = [{"auto_fields": 12, "review_fields": 0}, {"auto_fields": 9, "review_fields": 3}]
    assert approx(review_share_from_results(runs), 0.125)
    assert review_share_from_results([]) is None


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
