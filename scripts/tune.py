"""Tune LightGBM on log-rent with postal-code grouped CV, add conformal intervals, log to MLflow.

Run after scripts/train.py (reads data/processed/berlin_processed.csv).
Split by postal code: train / calibration / test, so every score is on unseen neighbourhoods.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import MODELS_DIR, PROCESSED_DATA_DIR
from berlinrentml.features.preprocessing import create_preprocessor
from berlinrentml.modeling.evaluation import calculate_metrics
from berlinrentml.modeling.splitting import GroupedSplit
from berlinrentml.modeling.training import save_model_artifacts
from berlinrentml.modeling.uncertainty import ConformalInterval

TARGET = "baseRent"
NUMERIC = ["livingSpace", "rooms", "floor", "building_age", "log_size", "rooms_per_m2", "amenity_score"]
CATEGORICAL = ["geo_plz", "geo_bln", "heatingType", "condition", "age_bin", "size_bin"]
ALPHA = 0.1
N_ITER = 40


def log_target(**params):
    return TransformedTargetRegressor(
        regressor=LGBMRegressor(random_state=42, verbose=-1, **params), func=np.log1p, inverse_func=np.expm1
    )


def main():
    df = pd.read_csv(PROCESSED_DATA_DIR / "berlin_processed.csv", dtype={"geo_plz": str})
    # Same test postal codes as scripts/train.py (seed 42, 20%); calibration is carved from the rest.
    X_rest, X_test, y_rest, y_test = GroupedSplit("geo_plz", 0.2, 42).split(df, TARGET)
    rest = X_rest.assign(**{TARGET: y_rest})
    X_train, X_cal, y_train, y_cal = GroupedSplit("geo_plz", 0.2, 7).split(rest, TARGET)
    print(f"train {len(X_train):,} | calibration {len(X_cal):,} | test {len(X_test):,} (disjoint postal codes)")

    prep = create_preprocessor(X_train, NUMERIC, CATEGORICAL)
    Xt, Xc, Xs = prep.fit_transform(X_train), prep.transform(X_cal), prep.transform(X_test)

    results = {}
    base = LGBMRegressor(n_estimators=100, max_depth=20, learning_rate=0.1, random_state=42, verbose=-1).fit(Xt, y_train)
    results["lightgbm_default"] = calculate_metrics(y_test, base.predict(Xs))
    results["lightgbm_log_target"] = calculate_metrics(y_test, log_target().fit(Xt, y_train).predict(Xs))

    print(f"\nTuning ({N_ITER} candidates, postal-code GroupKFold)...")
    search = RandomizedSearchCV(
        log_target(),
        {
            "regressor__n_estimators": randint(200, 900),
            "regressor__learning_rate": loguniform(0.01, 0.15),
            "regressor__num_leaves": randint(8, 64),
            "regressor__min_child_samples": randint(10, 60),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__subsample_freq": [1],
            "regressor__colsample_bytree": uniform(0.5, 0.5),
            "regressor__reg_lambda": loguniform(0.1, 30),
        },
        n_iter=N_ITER,
        cv=GroupKFold(5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    ).fit(Xt, y_train, groups=X_train["geo_plz"])
    best = search.best_estimator_
    results["lightgbm_log_tuned"] = calculate_metrics(y_test, best.predict(Xs))
    print(f"Best CV MAE: EUR {-search.best_score_:.2f}")

    # Cross-conformal: out-of-fold residuals on unseen postal codes over train + calibration rows.
    X_oof = sp.vstack([Xt, Xc]) if sp.issparse(Xt) else np.vstack([Xt, Xc])
    y_oof = np.concatenate([y_train, y_cal])
    g_oof = np.concatenate([X_train["geo_plz"], X_cal["geo_plz"]])
    oof = cross_val_predict(clone(best), X_oof, y_oof, groups=g_oof, cv=GroupKFold(5), n_jobs=-1)
    conf = ConformalInterval(ALPHA).fit(y_oof, oof)
    pred = best.predict(Xs)
    lo, hi = conf.predict(pred)
    coverage = conf.coverage(y_test.to_numpy(), pred)
    width = float(np.median(hi - lo))
    print(f"\n{'model':<22}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    for name, m in results.items():
        print(f"{name:<22}{m['mae']:>8.1f}{m['rmse']:>8.1f}{m['r2']:>8.3f}")
    print(f"\n{int((1 - ALPHA) * 100)}% interval: empirical coverage {coverage:.1%}, median width EUR {width:.0f}")

    save_model_artifacts(best, prep, NUMERIC + CATEGORICAL, "final_model", MODELS_DIR)
    joblib.dump(conf, MODELS_DIR / "final_model_conformal.joblib")
    pd.DataFrame(results).T.assign(coverage=coverage, interval_width=width).to_csv(MODELS_DIR / "tuning_results.csv")

    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("berlin-rent")
        with mlflow.start_run(run_name="lightgbm_log_tuned"):
            mlflow.log_params({k.replace("regressor__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metrics({f"{n}_{k}": v for n, m in results.items() for k, v in m.items() if k in ("mae", "rmse", "r2")})
            mlflow.log_metrics({"interval_coverage": coverage, "interval_median_width": width, "cv_mae": -search.best_score_})
        print("Logged run to MLflow (view: mlflow ui --backend-store-uri sqlite:///mlflow.db)")
    except ImportError:
        print("mlflow not installed, skipping tracking (pip install mlflow)")


if __name__ == "__main__":
    main()
