from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.random import RandomState
from scipy.optimize import Bounds, OptimizeResult, minimize


ArrayLike = Sequence[float] | np.ndarray


def _to_1d_float_array(values: ArrayLike, *, name: str, expected_size: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like, got shape {array.shape}.")
    if expected_size is not None and array.shape[0] != expected_size:
        raise ValueError(
            f"{name} must have length {expected_size}, got {array.shape[0]}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _normalize_feature_names(feature_names: Sequence[str]) -> list[str]:
    if not feature_names:
        raise ValueError("feature_names must be a non-empty sequence of strings.")
    normalized = [str(name) for name in feature_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("feature_names must be unique.")
    return normalized


def _normalize_bounds(
    bounds: Sequence[tuple[float | None, float | None]] | Bounds | None,
    n_features: int,
) -> list[tuple[float | None, float | None]] | None:
    if bounds is None:
        return None

    if isinstance(bounds, Bounds):
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        if lb.shape != (n_features,) or ub.shape != (n_features,):
            raise ValueError(
                f"Bounds arrays must have shape ({n_features},), got {lb.shape} and {ub.shape}."
            )
        normalized: list[tuple[float | None, float | None]] = []
        for low, high in zip(lb, ub):
            low_val = None if np.isneginf(low) else float(low)
            high_val = None if np.isposinf(high) else float(high)
            if low_val is not None and high_val is not None and low_val > high_val:
                raise ValueError(f"Invalid bounds interval: {(low_val, high_val)}")
            normalized.append((low_val, high_val))
        return normalized

    if len(bounds) != n_features:
        raise ValueError(f"bounds must have length {n_features}, got {len(bounds)}.")

    normalized = []
    for idx, bound in enumerate(bounds):
        if not isinstance(bound, (tuple, list)) or len(bound) != 2:
            raise ValueError(
                f"Each bounds entry must be a (lower, upper) pair; item {idx} is invalid."
            )
        low, high = bound
        low_val = None if low is None else float(low)
        high_val = None if high is None else float(high)
        if low_val is not None and not np.isfinite(low_val):
            raise ValueError(f"Lower bound at index {idx} must be finite or None.")
        if high_val is not None and not np.isfinite(high_val):
            raise ValueError(f"Upper bound at index {idx} must be finite or None.")
        if low_val is not None and high_val is not None and low_val > high_val:
            raise ValueError(f"Invalid bounds interval at index {idx}: {(low_val, high_val)}")
        normalized.append((low_val, high_val))
    return normalized


def _stabilized_inverse_covariance(
    reference_cov: ArrayLike,
    n_features: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    cov = np.asarray(reference_cov, dtype=float)
    if cov.shape != (n_features, n_features):
        raise ValueError(
            f"reference_cov must have shape ({n_features}, {n_features}), got {cov.shape}."
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError("reference_cov must contain only finite values.")

    cov = 0.5 * (cov + cov.T)
    scale = float(np.mean(np.diag(cov))) if np.any(np.diag(cov)) else 1.0
    cov = cov + np.eye(n_features) * max(jitter * max(scale, 1.0), jitter)
    return np.linalg.pinv(cov, hermitian=True)


def _predict_scalar(model: Any, x: np.ndarray) -> float:
    x_row = np.asarray(x, dtype=float).reshape(1, -1)

    if hasattr(model, "predict"):
        pred = model.predict(x_row)
    elif callable(model):
        pred = model(x_row)
    else:
        raise TypeError(
            "model must expose a .predict(X) method or be callable on a 2D array."
        )

    pred_array = np.asarray(pred, dtype=float).reshape(-1)
    if pred_array.size == 0:
        raise ValueError("Model prediction is empty.")
    if pred_array.size != 1:
        raise ValueError(
            f"Model prediction must return a single value for one sample, got shape {np.asarray(pred).shape}."
        )
    if not np.isfinite(pred_array[0]):
        raise ValueError("Model prediction is not finite.")
    return float(pred_array[0])


def _sample_start(
    rng: RandomState,
    n_features: int,
    *,
    bounds: list[tuple[float | None, float | None]] | None,
    reference_mean: np.ndarray | None,
    fallback_center: np.ndarray | None,
) -> np.ndarray:
    if bounds is not None:
        sampled = np.empty(n_features, dtype=float)
        center = reference_mean if reference_mean is not None else fallback_center
        for idx, (low, high) in enumerate(bounds):
            if low is not None and high is not None:
                sampled[idx] = rng.uniform(low, high)
            elif center is not None:
                sampled[idx] = center[idx] + rng.normal(scale=1.0)
                if low is not None:
                    sampled[idx] = max(sampled[idx], low)
                if high is not None:
                    sampled[idx] = min(sampled[idx], high)
            else:
                sampled[idx] = rng.normal(scale=1.0)
        return sampled

    if reference_mean is not None:
        return reference_mean + rng.normal(scale=1.0, size=n_features)
    if fallback_center is not None:
        return fallback_center + rng.normal(scale=1.0, size=n_features)
    return rng.normal(size=n_features)


def estimate_empirical_reference(
    X: ArrayLike,
    *,
    ddof: int = 1,
) -> dict[str, np.ndarray]:
    """
    Estimate empirical mean and covariance from a design matrix in model-input space.

    Parameters
    ----------
    X
        Array of shape (n_samples, n_features).
    ddof
        Delta degrees of freedom passed to `np.cov`.
    """
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_array.shape}.")
    if X_array.shape[0] < 2:
        raise ValueError("X must contain at least two rows to estimate covariance.")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X must contain only finite values.")

    mean = X_array.mean(axis=0)
    cov = np.cov(X_array, rowvar=False, ddof=ddof)
    cov = np.atleast_2d(cov).astype(float, copy=False)
    return {"mean": mean, "cov": cov}


class TorchRegressorAdapter:
    """
    Small adapter for torch models that consume already-processed numeric arrays.

    The wrapped model is expected to operate in the same feature space used by
    `inverse_yield`, typically the scaled numeric input space.
    """

    def __init__(self, model: Any, device: str = "cpu", dtype: str = "float32") -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

    def predict(self, X: ArrayLike) -> np.ndarray:
        import torch

        X_array = np.asarray(X, dtype=np.float32 if self.dtype == "float32" else np.float64)
        if X_array.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_array.shape}.")

        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X_array, device=self.device)
            preds = self.model(tensor).detach().cpu().numpy()
        return np.asarray(preds).reshape(-1)


def inverse_yield(
    model: Any,
    y_target: float,
    feature_names: Sequence[str],
    x0: ArrayLike | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | Bounds | None = None,
    reference_mean: ArrayLike | None = None,
    reference_cov: ArrayLike | None = None,
    lambda_mahal: float = 0.0,
    lambda_l2: float = 0.0,
    n_restarts: int = 10,
    random_state: int | RandomState = 42,
    method: str = "L-BFGS-B",
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Invert a trained regression model in the model's input feature space.

    Notes
    -----
    - This function optimizes only over the numeric feature vector `x`.
    - `x`, `x0`, `reference_mean`, `reference_cov`, and `bounds` must all be
      expressed in the same space expected by the model. If the predictive model
      was trained on standardized numeric inputs, inversion should also be done
      in that standardized space.
    - Crop is assumed fixed outside this function and is not optimized here.
    """
    feature_names = _normalize_feature_names(feature_names)
    n_features = len(feature_names)

    if not np.isfinite(y_target):
        raise ValueError("y_target must be finite.")
    y_target = float(y_target)

    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")
    if lambda_mahal < 0.0:
        raise ValueError("lambda_mahal must be non-negative.")
    if lambda_l2 < 0.0:
        raise ValueError("lambda_l2 must be non-negative.")

    normalized_bounds = _normalize_bounds(bounds, n_features)
    x0_array = None if x0 is None else _to_1d_float_array(x0, name="x0", expected_size=n_features)
    reference_mean_array = (
        None
        if reference_mean is None
        else _to_1d_float_array(reference_mean, name="reference_mean", expected_size=n_features)
    )

    if x0_array is not None and normalized_bounds is not None:
        clipped = []
        for value, (low, high) in zip(x0_array, normalized_bounds):
            if low is not None and value < low:
                value = low
            if high is not None and value > high:
                value = high
            clipped.append(value)
        x0_array = np.asarray(clipped, dtype=float)

    inv_cov = None
    if reference_cov is not None:
        inv_cov = _stabilized_inverse_covariance(reference_cov, n_features)

    reference_point = reference_mean_array
    if reference_point is None and x0_array is not None:
        reference_point = x0_array.copy()

    if isinstance(random_state, RandomState):
        rng = random_state
    else:
        rng = RandomState(random_state)

    scipy_bounds = None
    if normalized_bounds is not None:
        lb = [(-np.inf if low is None else low) for low, _ in normalized_bounds]
        ub = [(np.inf if high is None else high) for _, high in normalized_bounds]
        scipy_bounds = Bounds(lb, ub)

    def objective(x: np.ndarray) -> float:
        pred = _predict_scalar(model, x)
        value = (pred - y_target) ** 2

        if lambda_mahal > 0.0:
            if inv_cov is None:
                raise ValueError(
                    "reference_cov must be provided when lambda_mahal > 0."
                )
            if reference_mean_array is None:
                raise ValueError(
                    "reference_mean must be provided when lambda_mahal > 0."
                )
            delta = x - reference_mean_array
            value += lambda_mahal * float(delta.T @ inv_cov @ delta)

        if lambda_l2 > 0.0:
            if reference_point is None:
                raise ValueError(
                    "Provide x0 or reference_mean when lambda_l2 > 0."
                )
            delta_ref = x - reference_point
            value += lambda_l2 * float(np.dot(delta_ref, delta_ref))

        return float(value)

    starts: list[np.ndarray] = []
    if x0_array is not None:
        starts.append(x0_array.copy())

    while len(starts) < n_restarts:
        starts.append(
            _sample_start(
                rng,
                n_features,
                bounds=normalized_bounds,
                reference_mean=reference_mean_array,
                fallback_center=x0_array,
            )
        )

    restart_summaries: list[dict[str, Any]] = []
    best_result: OptimizeResult | None = None
    best_x: np.ndarray | None = None

    for restart_idx, start in enumerate(starts):
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method=method,
            bounds=scipy_bounds,
            options=dict(options or {}),
        )

        x_candidate = np.asarray(result.x, dtype=float).reshape(-1)
        y_candidate = _predict_scalar(model, x_candidate)
        summary = {
            "restart": restart_idx,
            "x0": np.asarray(start, dtype=float).copy(),
            "x_opt": x_candidate.copy(),
            "y_pred": y_candidate,
            "objective_value": float(result.fun),
            "success": bool(result.success),
            "status": int(getattr(result, "status", -1)),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
        }
        restart_summaries.append(summary)

        is_better = best_result is None or float(result.fun) < float(best_result.fun)
        if is_better:
            best_result = result
            best_x = x_candidate

    if best_result is None or best_x is None:
        raise RuntimeError("Inverse optimization failed before producing any result.")

    best_y_pred = _predict_scalar(model, best_x)
    return {
        "x_opt": best_x,
        "y_pred": best_y_pred,
        "objective_value": float(best_result.fun),
        "success": bool(best_result.success),
        "message": str(best_result.message),
        "restart_summaries": restart_summaries,
        "feature_names": feature_names,
        "y_target": y_target,
        "n_restarts": n_restarts,
    }


__all__ = [
    "TorchRegressorAdapter",
    "estimate_empirical_reference",
    "inverse_yield",
]
