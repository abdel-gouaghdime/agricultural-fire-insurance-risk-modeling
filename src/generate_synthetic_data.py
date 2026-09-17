"""Generate a fully synthetic agricultural fire-insurance portfolio.

The data reproduce only generic actuarial relationships. They do not contain
records from Credit Agricole Assurances or the Challenge Data competition.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "synthetic_agricultural_portfolio.csv"


def generate_portfolio(n_policies: int = 8_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    activity = rng.choice(
        ["crop_farming", "livestock", "mixed_farming", "viticulture"],
        size=n_policies,
        p=[0.30, 0.28, 0.32, 0.10],
    )
    region_risk = rng.choice(["low", "medium", "high"], n_policies, p=[0.35, 0.45, 0.20])
    building_material = rng.choice(["masonry", "mixed", "wood"], n_policies, p=[0.55, 0.30, 0.15])
    exposure = rng.uniform(0.4, 1.0, n_policies).round(3)
    surface_m2 = np.clip(rng.lognormal(7.1, 0.65, n_policies), 150, 25_000).round(0)
    insured_capital = np.clip(
        surface_m2 * rng.uniform(210, 520, n_policies) + rng.lognormal(11.0, 0.7, n_policies),
        50_000,
        8_000_000,
    ).round(0)
    building_age = rng.integers(1, 81, n_policies)
    previous_claims = np.clip(rng.poisson(0.22, n_policies), 0, 4)
    fire_protection = rng.binomial(1, 0.68, n_policies)
    fire_station_km = np.clip(rng.gamma(2.2, 4.0, n_policies), 0.5, 40).round(1)

    activity_effect = pd.Series(activity).map(
        {"crop_farming": 0.05, "livestock": 0.20, "mixed_farming": 0.28, "viticulture": -0.08}
    ).to_numpy()
    region_effect = pd.Series(region_risk).map({"low": -0.25, "medium": 0.0, "high": 0.35}).to_numpy()
    material_effect = pd.Series(building_material).map({"masonry": -0.15, "mixed": 0.05, "wood": 0.42}).to_numpy()

    log_frequency = (
        -3.15
        + activity_effect
        + region_effect
        + material_effect
        + 0.24 * previous_claims
        - 0.32 * fire_protection
        + 0.012 * fire_station_km
        + 0.0028 * building_age
        + 0.000018 * surface_m2
    )
    annual_frequency = np.exp(log_frequency)
    claim_count = rng.poisson(annual_frequency * exposure)

    log_severity = (
        9.15
        + 0.28 * material_effect
        + 0.18 * region_effect
        + 0.17 * np.log1p(surface_m2 / 1_000)
        + 0.22 * np.log1p(insured_capital / 500_000)
        - 0.12 * fire_protection
        + 0.008 * fire_station_km
    )
    expected_severity = np.exp(log_severity)
    total_claim_cost = np.zeros(n_policies)
    positive = claim_count > 0
    total_claim_cost[positive] = rng.gamma(
        shape=2.0 * claim_count[positive],
        scale=expected_severity[positive] / 2.0,
    )
    average_claim_cost = np.divide(
        total_claim_cost,
        claim_count,
        out=np.zeros_like(total_claim_cost),
        where=claim_count > 0,
    )

    return pd.DataFrame(
        {
            "policy_id": [f"SYN-{i:06d}" for i in range(1, n_policies + 1)],
            "exposure_years": exposure,
            "activity": activity,
            "region_risk": region_risk,
            "building_material": building_material,
            "surface_m2": surface_m2,
            "insured_capital_eur": insured_capital,
            "building_age_years": building_age,
            "previous_claims": previous_claims,
            "fire_protection": fire_protection,
            "fire_station_distance_km": fire_station_km,
            "claim_count": claim_count,
            "average_claim_cost_eur": average_claim_cost.round(2),
            "total_claim_cost_eur": total_claim_cost.round(2),
        }
    )


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    portfolio = generate_portfolio()
    portfolio.to_csv(OUTPUT, index=False)
    print(f"Saved {len(portfolio):,} synthetic policies to {OUTPUT}")
