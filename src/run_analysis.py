"""Run a reproducible frequency-severity benchmark on synthetic data."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_agricultural_portfolio.csv"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

FEATURES = [
    "activity", "region_risk", "building_material", "surface_m2",
    "insured_capital_eur", "building_age_years", "previous_claims",
    "fire_protection", "fire_station_distance_km",
]
CATEGORICAL = ["activity", "region_risk", "building_material"]
NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]


def preprocessing(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.25, random_state=42)

    x_train, x_test = train[FEATURES], test[FEATURES]
    train_rate = train["claim_count"] / train["exposure_years"]
    test_rate = test["claim_count"] / test["exposure_years"]

    frequency_models = {
        "GLM Poisson": Pipeline([
            ("preprocess", preprocessing(scale=True)),
            ("model", PoissonRegressor(alpha=0.1, max_iter=1_000)),
        ]),
        "Gradient Boosting Poisson": Pipeline([
            ("preprocess", preprocessing(scale=False)),
            ("model", HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.06, max_iter=220,
                max_leaf_nodes=15, l2_regularization=1.0, random_state=42,
            )),
        ]),
    }

    freq_predictions = {}
    rows = []
    for name, model in frequency_models.items():
        model.fit(x_train, train_rate, model__sample_weight=train["exposure_years"])
        prediction = np.clip(model.predict(x_test), 1e-6, None)
        freq_predictions[name] = prediction
        rows.append({
            "component": "Frequency",
            "model": name,
            "MAE": mean_absolute_error(test_rate, prediction),
            "deviance": mean_poisson_deviance(test_rate, prediction),
        })

    claims = train[train["claim_count"] > 0].copy()
    claims_train, claims_valid = train_test_split(claims, test_size=0.30, random_state=42)
    severity_models = {
        "GLM Tweedie": Pipeline([
            ("preprocess", preprocessing(scale=True)),
            ("model", TweedieRegressor(power=1.5, alpha=0.2, link="log", max_iter=2_000)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessing(scale=False)),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=8, max_features=0.8,
                n_jobs=-1, random_state=42,
            )),
        ]),
    }

    severity_validation_predictions = {}
    for name, model in severity_models.items():
        model.fit(
            claims_train[FEATURES], claims_train["average_claim_cost_eur"],
            model__sample_weight=claims_train["claim_count"],
        )
        prediction = np.clip(model.predict(claims_valid[FEATURES]), 1.0, None)
        severity_validation_predictions[name] = prediction
        rows.append({
            "component": "Severity",
            "model": name,
            "MAE": mean_absolute_error(claims_valid["average_claim_cost_eur"], prediction),
            "deviance": mean_tweedie_deviance(
                claims_valid["average_claim_cost_eur"], prediction, power=1.5
            ),
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "model_metrics.csv", index=False)
    best_frequency_name = results.loc[results["component"].eq("Frequency")].sort_values("deviance").iloc[0]["model"]
    best_severity_name = results.loc[results["component"].eq("Severity")].sort_values("deviance").iloc[0]["model"]

    best_severity = severity_models[best_severity_name]
    best_severity.fit(
        claims[FEATURES], claims["average_claim_cost_eur"],
        model__sample_weight=claims["claim_count"],
    )
    severity_test = np.clip(best_severity.predict(x_test), 1.0, None)
    pure_premium_prediction = freq_predictions[best_frequency_name] * severity_test
    observed_pure_premium = test["total_claim_cost_eur"] / test["exposure_years"]
    premium_mae = mean_absolute_error(observed_pure_premium, pure_premium_prediction)
    observed_total_cost = float(test["total_claim_cost_eur"].sum())
    predicted_total_cost = float((pure_premium_prediction * test["exposure_years"]).sum())

    predictions = test[["policy_id", "exposure_years", "claim_count", "total_claim_cost_eur"]].copy()
    predictions["frequency_prediction"] = freq_predictions[best_frequency_name]
    predictions["severity_prediction_eur"] = severity_test
    predictions["pure_premium_prediction_eur"] = pure_premium_prediction
    predictions.to_csv(RESULTS / "test_predictions.csv", index=False)

    summary = {
        "dataset": "synthetic demonstration data",
        "n_policies": int(len(df)),
        "train_policies": int(len(train)),
        "test_policies": int(len(test)),
        "observed_claim_rate": float(df["claim_count"].sum() / df["exposure_years"].sum()),
        "best_frequency_model": best_frequency_name,
        "best_severity_model": best_severity_name,
        "test_pure_premium_mae_eur": float(premium_mae),
        "observed_test_total_cost_eur": observed_total_cost,
        "predicted_test_total_cost_eur": predicted_total_cost,
        "portfolio_calibration_ratio": predicted_total_cost / observed_total_cost,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(df["claim_count"], discrete=True, ax=axes[0], color="#1f5a7a")
    axes[0].set(title="Distribution of fire claim counts", xlabel="Claim count", ylabel="Policies")
    positive_costs = df.loc[df["total_claim_cost_eur"] > 0, "total_claim_cost_eur"]
    sns.histplot(positive_costs.clip(upper=positive_costs.quantile(0.99)), bins=35, ax=axes[1], color="#d77a2b")
    axes[1].set(title="Positive claim costs (capped at p99)", xlabel="Total cost (€)", ylabel="Policies")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    chart = results.copy()
    chart["deviance_index"] = chart.groupby("component")["deviance"].transform(lambda s: 100 * s / s.min())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=chart, x="deviance_index", y="model", hue="component", ax=ax)
    ax.axvline(100, color="black", linestyle="--", linewidth=1)
    ax.set(title="Relative model deviance (best in each component = 100)", xlabel="Relative deviance index", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    best_frequency = frequency_models[best_frequency_name]
    importance = permutation_importance(
        best_frequency, x_test, test_rate, n_repeats=8,
        random_state=42, scoring="neg_mean_poisson_deviance",
    )
    imp = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "frequency_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp.head(10), x="importance", y="feature", color="#1f5a7a", ax=ax)
    ax.set(title=f"Frequency drivers — {best_frequency_name}", xlabel="Permutation importance", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES / "frequency_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
