"""Small shared measurement functions; missing observations are not successes."""
import math


def sampled_age_summary(ages_ms, missing):
    if missing < 0 or any(not math.isfinite(value) or value < 0 for value in ages_ms):
        raise ValueError("Invalid age observations")
    total = len(ages_ms) + missing
    ordered = sorted(ages_ms)
    index = math.ceil(.95 * total) - 1
    p95 = ordered[index] if total and index < len(ordered) else None
    return {"latest_age_p95_ms": p95, "p95_has_no_finite_observation": p95 is None,
            "latest_age_max_ms": max(ordered) if ordered else None,
            "no_result_fraction": missing / total if total else 1,
            "stale_over_200ms_fraction": (sum(age > 200 for age in ordered) + missing) / total if total else 1,
            "sample_count": total, "percentile_method": "nearest-rank; missing samples rank after all finite ages"}
