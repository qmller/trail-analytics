"""Pace models: seconds per metre as a function of block features."""

import numpy as np
import pandas as pd

FEATURES = ["grade", "grade_prev", "grade_next", "ele", "effort"]


def features(blocks: pd.DataFrame, hr_rel=None) -> pd.DataFrame:
    g = blocks["grade"].to_numpy()
    X = pd.DataFrame({
        "grade": g,
        "grade_prev": np.r_[g[:1], g[:-1]],
        "grade_next": np.r_[g[1:], g[-1:]],
        "ele": blocks["ele"].to_numpy(),
        "effort": blocks["effort"].to_numpy(),
    })
    if hr_rel is not None:
        X["hr_rel"] = np.broadcast_to(np.asarray(hr_rel, dtype=float), (len(X),))
    return X


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = np.cumsum(w) / w.sum()
    return np.interp(q, cw, v)


class PaceModel:
    name = ""
    use_hr = False

    def fit(self, X: pd.DataFrame, spm: np.ndarray, w: np.ndarray) -> "PaceModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class BinLookupModel(PaceModel):
    """Time-weighted pace per grade bin, linearly interpolated (fixed version of the original approach)."""

    name = "bins"

    def __init__(self, bin_size=1.0, min_blocks=3, **_):
        self.bin_size = bin_size
        self.min_blocks = min_blocks

    def fit(self, X, spm, w):
        edges = np.arange(-60.0, 60.0 + self.bin_size, self.bin_size)
        idx = np.clip(np.digitize(X["grade"], edges) - 1, 0, len(edges) - 2)
        num = np.bincount(idx, w * spm, len(edges) - 1)
        den = np.bincount(idx, w, len(edges) - 1)
        cnt = np.bincount(idx, minlength=len(edges) - 1)
        ok = (cnt >= self.min_blocks) & (den > 0)
        self.grades = ((edges[:-1] + edges[1:]) / 2)[ok]
        self.spm = num[ok] / den[ok]
        return self

    def predict(self, X):
        return np.interp(X["grade"], self.grades, self.spm)


class CurveModel(PaceModel):
    """log(pace) = poly(grade) [+ k * effort-km (fatigue)] [+ b * (HR/LTHR - 1)], weighted least squares.

    Beyond the 1st/99th grade percentiles of the data the curve is extended linearly.
    """

    name = "curve"

    def __init__(self, degree=4, fatigue=True, **_):
        self.degree = degree
        self.fatigue = fatigue

    def _design(self, X):
        g = X["grade"].to_numpy() / 100.0
        gc = np.clip(g, self.lo, self.hi)
        cols = [gc**i for i in range(self.degree + 1)]
        if self.use_fatigue:
            cols.append(X["effort"].to_numpy())
        if self.use_hr:
            cols.append(X["hr_rel"].to_numpy() - 1.0)
        return np.column_stack(cols), g, gc

    def _solve(self, X, y, w):
        A, _, _ = self._design(X)
        sw = np.sqrt(w)
        return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]

    def fit(self, X, spm, w):
        g = X["grade"].to_numpy() / 100.0
        self.lo, self.hi = weighted_quantile(g, w, [0.01, 0.99])
        y = np.log(spm)
        self.use_fatigue = self.fatigue
        self.use_hr = "hr_rel" in X
        coef = self._solve(X, y, w)
        # A negative fatigue or positive HR coefficient is confounding, not physiology: drop the term.
        if self.use_hr and coef[-1] > 0:
            self.use_hr = False
            coef = self._solve(X, y, w)
        if self.use_fatigue and coef[self.degree + 1] < 0:
            self.use_fatigue = False
            coef = self._solve(X, y, w)
        self.coef = coef
        poly = np.polynomial.Polynomial(coef[: self.degree + 1])
        d = poly.deriv()
        self.slope_lo = min(d(self.lo), 0.0)
        self.slope_hi = max(d(self.hi), 0.0)
        return self

    @property
    def fatigue_per_effort_km(self) -> float:
        return float(self.coef[self.degree + 1]) if self.use_fatigue else 0.0

    def predict(self, X):
        A, g, gc = self._design(X)
        logp = A @ self.coef
        logp += np.where(g < self.lo, self.slope_lo, self.slope_hi) * (g - gc)
        return np.exp(logp)


class GBMModel(PaceModel):
    """LightGBM Huber (outlier-robust) regression on log(pace); monotone in effort-km (fatigue) and HR."""

    name = "gbm"

    def __init__(self, fatigue=True, n_estimators=400, **_):
        self.fatigue = fatigue
        self.n_estimators = n_estimators

    def fit(self, X, spm, w):
        from lightgbm import LGBMRegressor

        self.cols = [c for c in X.columns if self.fatigue or c != "effort"]
        self.use_hr = "hr_rel" in self.cols
        mono = {"effort": 1, "hr_rel": -1}
        self.model = LGBMRegressor(
            objective="huber",
            alpha=0.3,
            n_estimators=self.n_estimators,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            monotone_constraints=[mono.get(c, 0) for c in self.cols],
            verbose=-1,
        )
        self.model.fit(X[self.cols], np.log(spm), sample_weight=w)
        return self

    def predict(self, X):
        return np.exp(self.model.predict(X[self.cols]))


MODELS = {m.name: m for m in (BinLookupModel, CurveModel, GBMModel)}


def make_model(name: str, fatigue=True) -> PaceModel:
    try:
        return MODELS[name](fatigue=fatigue)
    except KeyError:
        raise ValueError(f"unknown model {name!r}, choose from {sorted(MODELS)}") from None
