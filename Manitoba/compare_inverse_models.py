from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from inverse_modeling import estimate_empirical_reference, inverse_yield


DATA_PATH = Path(__file__).resolve().parent / "prepared" / "yields_weather_by_crop.csv"
OUTDIR = Path(__file__).resolve().parent / "prepared"
OUTDIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CROP = "RED SPRING WHEAT"
TEST_SIZE = 0.2
TARGET_QUANTILES = {"low": 0.25, "median": 0.5, "high": 0.75}
INVERSE_LAMBDA_L2 = 1e-3
INVERSE_RESTARTS = 8


def gp_param_grid() -> list[dict[str, list[object]]]:
    return [
        {
            "alpha": [1e-6, 1e-4, 1e-2],
            "kernel_name": ["matern_1p5"],
            "kernel": [
                ConstantKernel(1.0, (1e-2, 1e2))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
            ],
        },
        {
            "alpha": [1e-6, 1e-4, 1e-2],
            "kernel_name": ["matern_2p5"],
            "kernel": [
                ConstantKernel(1.0, (1e-2, 1e2))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
            ],
        },
        {
            "alpha": [1e-6, 1e-4, 1e-2],
            "kernel_name": ["rbf"],
            "kernel": [
                ConstantKernel(1.0, (1e-2, 1e2))
                * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
            ],
        },
    ]


def load_crop_frame(path: Path, crop: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    climate_feature_cols = [c for c in df.columns if re.match(r"^\d{2}_Q[12]_", c)]
    fixed_crop_df = (
        df.loc[df["Crop"] == crop, ["Mean_Yield", "Crop"] + climate_feature_cols]
        .dropna()
        .reset_index(drop=True)
    )
    return fixed_crop_df, climate_feature_cols


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tune_gp_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, float | object], pd.DataFrame]:
    """
    Tune GP hyperparameters using train/val split validation.
    
    Searches over alpha values and kernel configurations.
    Returns the best hyperparameters based on validation RMSE.
    """
    # Split training data for hyperparameter tuning
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )
    
    best_key = None
    best_params = None
    tuning_rows: list[dict[str, float | str]] = []
    
    print("  Tuning GP hyperparameters...")
    for params in ParameterGrid(gp_param_grid()):
        gp = GaussianProcessRegressor(
            kernel=params["kernel"],
            alpha=params["alpha"],
            normalize_y=True,
            n_restarts_optimizer=10,  # Enable optimizer to find good hyperparameters
            random_state=RANDOM_STATE,
        )
        gp.fit(X_tune, y_tune)
        y_val_pred = gp.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        val_r2 = float(r2_score(y_val, y_val_pred))
        row = {
            "alpha": float(params["alpha"]),
            "kernel_name": str(params["kernel_name"]),
            "val_rmse": val_rmse,
            "val_mae": float(mean_absolute_error(y_val, y_val_pred)),
            "val_r2": val_r2,
            "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
            "optimized_kernel": str(gp.kernel_),
        }
        tuning_rows.append(row)

        rank_key = (val_rmse, -val_r2)
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_params = params

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["val_rmse", "val_r2"],
        ascending=[True, False],
    ).reset_index(drop=True)
    best_row = tuning_df.iloc[0]
    print(f"  Best validation RMSE: {best_row['val_rmse']:.4f}")
    print(f"  Best validation R2: {best_row['val_r2']:.4f}")
    print(f"  Best alpha: {best_params['alpha']}")
    print(f"  Best kernel family: {best_params['kernel_name']}")
    return best_params, tuning_df


def build_models(gp_hyperparams: dict[str, float | object] | None = None) -> dict[str, object]:
    if gp_hyperparams is None:
        # Default hyperparameters
        gp_hyperparams = {
            "alpha": 1e-4,
            "kernel": (
                ConstantKernel(1.0, (1e-2, 1e2))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))
            ),
        }
    
    return {
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GP": GaussianProcessRegressor(
            kernel=gp_hyperparams["kernel"],
            alpha=gp_hyperparams["alpha"],
            normalize_y=True,
            n_restarts_optimizer=10,  # Enable optimizer during final training
            random_state=RANDOM_STATE,
        ),
    }


def compare_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_crop_df, climate_feature_cols = load_crop_frame(DATA_PATH, CROP)

    X = fixed_crop_df[climate_feature_cols].copy()
    y = fixed_crop_df["Mean_Yield"].to_numpy(dtype=float)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Tune GP hyperparameters on training data
    print("\nTuning Gaussian Process hyperparameters...")
    gp_hyperparams, gp_tuning_df = tune_gp_hyperparameters(X_train, y_train)

    reference = estimate_empirical_reference(X_train)
    reference_mean = reference["mean"]
    reference_cov = reference["cov"]

    lower = np.quantile(X_train, 0.01, axis=0)
    upper = np.quantile(X_train, 0.99, axis=0)
    bounds = list(zip(lower, upper))
    train_quantiles = pd.Series(y_train).quantile(list(TARGET_QUANTILES.values()))
    nn = NearestNeighbors(n_neighbors=1).fit(X_train)

    comparison_rows: list[dict[str, float | str | bool]] = []
    inverse_rows: list[dict[str, float | str | bool]] = []

    for model_name, model in build_models(gp_hyperparams).items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_metrics = regression_metrics(y_train, y_train_pred)
        test_metrics = regression_metrics(y_test, y_test_pred)

        model_row: dict[str, float | str | bool] = {
            "model": model_name,
            "crop": CROP,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        for split_name, split_metrics in [("train", train_metrics), ("test", test_metrics)]:
            for metric_name, metric_value in split_metrics.items():
                model_row[f"{split_name}_{metric_name}"] = metric_value

        inverse_abs_errors = []
        inverse_bound_hits = []
        inverse_neighbor_distances = []
        inverse_l2_from_mean = []

        for scenario, q in TARGET_QUANTILES.items():
            y_target = float(train_quantiles.loc[q])
            result = inverse_yield(
                model=model,
                y_target=y_target,
                feature_names=climate_feature_cols,
                bounds=bounds,
                reference_mean=reference_mean,
                reference_cov=reference_cov,
                lambda_mahal=0.0,
                lambda_l2=INVERSE_LAMBDA_L2,
                n_restarts=INVERSE_RESTARTS,
                random_state=RANDOM_STATE,
            )

            x_opt = result["x_opt"]
            nearest_distance = float(nn.kneighbors(x_opt.reshape(1, -1), return_distance=True)[0][0, 0])
            bound_hits = int(
                np.sum(
                    np.isclose(x_opt, lower, atol=1e-6)
                    | np.isclose(x_opt, upper, atol=1e-6)
                )
            )
            l2_from_mean = float(np.linalg.norm(x_opt - reference_mean))
            abs_error = float(abs(result["y_pred"] - y_target))

            inverse_abs_errors.append(abs_error)
            inverse_bound_hits.append(bound_hits)
            inverse_neighbor_distances.append(nearest_distance)
            inverse_l2_from_mean.append(l2_from_mean)

            inverse_rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "y_target": y_target,
                    "y_pred": float(result["y_pred"]),
                    "abs_error": abs_error,
                    "objective_value": float(result["objective_value"]),
                    "success": bool(result["success"]),
                    "l2_from_reference_mean": l2_from_mean,
                    "nearest_train_distance": nearest_distance,
                    "n_features_at_bounds": bound_hits,
                }
            )

        model_row["inverse_mean_abs_error"] = float(np.mean(inverse_abs_errors))
        model_row["inverse_max_abs_error"] = float(np.max(inverse_abs_errors))
        model_row["inverse_mean_l2_from_mean"] = float(np.mean(inverse_l2_from_mean))
        model_row["inverse_mean_nearest_train_distance"] = float(np.mean(inverse_neighbor_distances))
        model_row["inverse_mean_features_at_bounds"] = float(np.mean(inverse_bound_hits))
        comparison_rows.append(model_row)

    summary_df = pd.DataFrame(comparison_rows).sort_values(
        ["inverse_mean_abs_error", "test_rmse", "test_r2"],
        ascending=[True, True, False],
    )
    inverse_df = pd.DataFrame(inverse_rows).sort_values(["model", "scenario"])
    return summary_df, inverse_df, gp_tuning_df


def main() -> None:
    summary_df, inverse_df, gp_tuning_df = compare_models()
    summary_path = OUTDIR / "inverse_model_comparison_summary.csv"
    inverse_path = OUTDIR / "inverse_model_comparison_by_target.csv"
    gp_tuning_path = OUTDIR / "gp_hyperparameter_tuning_results.csv"

    summary_df.to_csv(summary_path, index=False)
    inverse_df.to_csv(inverse_path, index=False)
    gp_tuning_df.to_csv(gp_tuning_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("Model comparison summary")
    print(summary_df.round(4).to_string(index=False))
    print()
    print("GP hyperparameter tuning results")
    print(gp_tuning_df.round(4).to_string(index=False))
    print()
    print("Inverse target details")
    print(inverse_df.round(4).to_string(index=False))
    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Saved target-level results to: {inverse_path}")
    print(f"Saved GP tuning results to: {gp_tuning_path}")


if __name__ == "__main__":
    main()
