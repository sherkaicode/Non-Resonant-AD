"""
gen_validation.py
==================

A validation suite for data-driven generative background models used in
non-resonant / weakly-supervised anomaly detection at colliders -- e.g. the
"Generate" (CATHODE-style normalizing-flow) method of Bai, Mastandrea &
Nachman, arXiv:2311.12924.

It compares a held-out real-data sample (never used in training) against
generator (GEN) output on:

  1. check_bounds            -- physical range sanity (positivity, [0,1], ...)
  2. compare_moments          -- weighted mean/std/skew/kurtosis + bootstrap errors
  3. compare_marginals        -- weighted KS test, Wasserstein distance, tail
                                  ratio, and ratio/pull plots, per feature
  4. compare_correlations     -- correlation matrices + corner plot (joint
                                  structure -- catches things 1D checks miss)
  5. classifier_two_sample_test -- train a classifier to separate GEN from
                                  data; AUC ~ 0.5 means "indistinguishable"
  6. weight_diagnostics       -- effective sample size / extreme-weight
                                  fraction of the GEN importance weights
  7. nearest_neighbor_check   -- local-density / coverage check against data

`run_full_validation(...)` runs all of the above, saves plots, and returns
(and prints) a single structured report.

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn (all standard).

Example
-------
    from gen_validation import run_full_validation

    report = run_full_validation(
        data_events=held_out_data_df,      # real data, NOT used in training
        gen_events=gen_df,                 # output of the GEN model
        weight_col="weight",               # w(m) importance weight column
        feature_list=["m_jj", "tau21_lead", "tau32_lead",
                      "tau21_sublead", "tau32_sublead"],
        bounds={"m_jj": (0, None),
                "tau21_lead": (0, 1), "tau32_lead": (0, 1),
                "tau21_sublead": (0, 1), "tau32_sublead": (0, 1)},
        out_dir="gen_validation_plots",
    )
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors


# ==========================================================================
# Small numerical helpers
# ==========================================================================

def _get_weights(df: pd.DataFrame, weight_col: Optional[str]) -> np.ndarray:
    """Return a weight array for df; defaults to all-ones if absent."""
    if weight_col is not None and weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=float)
        if np.any(w < 0):
            warnings.warn(f"Negative weights found in column '{weight_col}'; "
                           f"{np.sum(w < 0)} events affected.")
        w = np.abs(w)

    else:
        w = np.ones(len(df))
    return w


def _weighted_moments(x: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    w = w / w.sum()
    mean = np.sum(w * x)
    var = np.sum(w * (x - mean) ** 2)
    std = np.sqrt(max(var, 0.0))
    if std > 0:
        skew = np.sum(w * ((x - mean) / std) ** 3)
        kurt = np.sum(w * ((x - mean) / std) ** 4) - 3.0
    else:
        skew, kurt = np.nan, np.nan
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


def _bootstrap_weighted_mean_err(x: np.ndarray, w: np.ndarray,
                                  n_boot: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = np.average(x[idx], weights=w[idx])
    return float(np.std(means))


def _weighted_ks_statistic(x1, w1, x2, w2) -> Tuple[float, float]:
    """Two-sample weighted KS statistic with an asymptotic p-value
    (uses the effective sample size of each weighted sample)."""
    data_all = np.concatenate([x1, x2])
    order = np.argsort(data_all)

    cw1 = np.concatenate([w1, np.zeros_like(w2)])[order]
    cw2 = np.concatenate([np.zeros_like(w1), w2])[order]

    cdf1 = np.cumsum(cw1) / w1.sum()
    cdf2 = np.cumsum(cw2) / w2.sum()
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    n_eff = _effective_sample_size(w1)
    m_eff = _effective_sample_size(w2)
    en = np.sqrt(n_eff * m_eff / (n_eff + m_eff))
    lam = (en + 0.12 + 0.11 / en) * d_stat
    p_value = 2.0 * sum((-1.0) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)
                         for k in range(1, 101))
    return d_stat, float(np.clip(p_value, 0.0, 1.0))


def _weighted_quantile(values: np.ndarray, quantiles: Sequence[float],
                        weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return np.interp(quantiles, cw, v)


def _effective_sample_size(w: np.ndarray) -> float:
    """Kish's effective sample size: (sum w)^2 / sum(w^2)."""
    s2 = np.sum(w ** 2)
    return float((w.sum() ** 2) / s2) if s2 > 0 else 0.0


def _safe_makedirs(path: Optional[str]):
    if path:
        os.makedirs(path, exist_ok=True)


# ==========================================================================
# 1. Physical sanity bounds
# ==========================================================================

def check_bounds(gen_events: pd.DataFrame, feature_list: List[str],
                  bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
                  ) -> Dict[str, dict]:
    """
    Check generated features against physical bounds, e.g. m_jj >= 0 or
    N-subjettiness ratios in [0, 1]. `bounds` maps feature -> (lo, hi),
    either side may be None to skip that side. Features with no entry in
    `bounds` are skipped (no assumption is made about their range).

    Returns a dict per feature with counts/fractions of violating events
    and the most extreme offending value.
    """
    bounds = bounds or {}
    results = {}
    for feat in feature_list:
        if feat not in bounds:
            continue
        lo, hi = bounds[feat]
        x = gen_events[feat].to_numpy(dtype=float)
        n_total = len(x)
        n_nan = int(np.isnan(x).sum())
        n_below = int(np.sum(x < lo)) if lo is not None else 0
        n_above = int(np.sum(x > hi)) if hi is not None else 0
        results[feat] = {
            "n_total": n_total,
            "n_nan_or_inf": n_nan + int(np.isinf(x).sum()),
            "n_below_lo": n_below,
            "frac_below_lo": n_below / n_total if n_total else np.nan,
            "n_above_hi": n_above,
            "frac_above_hi": n_above / n_total if n_total else np.nan,
            "min_value": float(np.nanmin(x)) if n_total else np.nan,
            "max_value": float(np.nanmax(x)) if n_total else np.nan,
            "bounds": (lo, hi),
        }
    return results


# ==========================================================================
# 2. Weighted moments comparison
# ==========================================================================

def compare_moments(data_events: pd.DataFrame, gen_events: pd.DataFrame,
                     weight_col: Optional[str], feature_list: List[str],
                     n_boot: int = 200) -> pd.DataFrame:
    """Weighted mean/std/skew/kurtosis for data vs. gen, with a bootstrap
    error on the mean and a pull = (mean_gen - mean_data) / sigma_data."""
    w_data = _get_weights(data_events, weight_col)
    w_gen = _get_weights(gen_events, weight_col)

    rows = []
    for feat in feature_list:
        xd = data_events[feat].to_numpy(dtype=float)
        xg = gen_events[feat].to_numpy(dtype=float)
        md = _weighted_moments(xd, w_data)
        mg = _weighted_moments(xg, w_gen)
        err_d = _bootstrap_weighted_mean_err(xd, w_data, n_boot=n_boot)
        err_g = _bootstrap_weighted_mean_err(xg, w_gen, n_boot=n_boot)
        pull = (mg["mean"] - md["mean"]) / err_d if err_d > 0 else np.nan
        rows.append({
            "feature": feat,
            "mean_data": md["mean"], "mean_gen": mg["mean"],
            "mean_err_data": err_d, "mean_err_gen": err_g,
            "mean_pull_sigma": pull,
            "std_data": md["std"], "std_gen": mg["std"],
            "skew_data": md["skew"], "skew_gen": mg["skew"],
            "kurtosis_data": md["kurtosis"], "kurtosis_gen": mg["kurtosis"],
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 3. Marginal shape agreement (KS, Wasserstein, tails, ratio plots)
# ==========================================================================

def compare_marginals(data_events: pd.DataFrame, gen_events: pd.DataFrame,
                       weight_col: Optional[str], feature_list: List[str],
                       bins: int = 40, tail_quantile: float = 0.95,
                       out_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Per feature: weighted KS statistic + p-value, weighted 1-Wasserstein
    distance, and a tail-region check (fraction of events above the
    `tail_quantile` quantile of the DATA distribution, data vs. gen).
    Also saves a histogram + ratio ("pull") plot per feature if out_dir given.
    """
    _safe_makedirs(out_dir)
    w_data = _get_weights(data_events, weight_col)
    w_gen = _get_weights(gen_events, weight_col)

    rows = []
    for feat in feature_list:
        xd = data_events[feat].to_numpy(dtype=float)
        xg = gen_events[feat].to_numpy(dtype=float)

        ks_stat, ks_p = _weighted_ks_statistic(xd, w_data, xg, w_gen)
        wass = stats.wasserstein_distance(xd, xg, u_weights=w_data, v_weights=w_gen)
        # normalize by the data spread so distances are comparable across features
        spread = np.std(xd) if np.std(xd) > 0 else 1.0
        wass_norm = wass / spread

        tail_cut = _weighted_quantile(xd, [tail_quantile], w_data)[0]
        tail_frac_data = np.sum(w_data[xd > tail_cut]) / w_data.sum()
        tail_frac_gen = np.sum(w_gen[xg > tail_cut]) / w_gen.sum()

        rows.append({
            "feature": feat,
            "ks_stat": ks_stat, "ks_pvalue": ks_p,
            "wasserstein": wass, "wasserstein_norm": wass_norm,
            f"tail_frac_data_(>q{tail_quantile})": tail_frac_data,
            f"tail_frac_gen_(>q{tail_quantile})": tail_frac_gen,
            "tail_ratio_gen_over_data": (tail_frac_gen / tail_frac_data
                                         if tail_frac_data > 0 else np.nan),
        })

        if out_dir:
            _plot_marginal(xd, w_data, xg, w_gen, feat, bins, out_dir)

    return pd.DataFrame(rows)


def _plot_marginal(xd, wd, xg, wg, feat, bins, out_dir):
    lo = min(xd.min(), xg.min())
    hi = max(xd.max(), xg.max())
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hd, _ = np.histogram(xd, bins=edges, weights=wd)
    hg, _ = np.histogram(xg, bins=edges, weights=wg)
    # Poisson-like errors accounting for weights (sum of w^2 per bin)
    hd_err2, _ = np.histogram(xd, bins=edges, weights=wd ** 2)
    hg_err2, _ = np.histogram(xg, bins=edges, weights=wg ** 2)
    hd_err, hg_err = np.sqrt(hd_err2), np.sqrt(hg_err2)

    # normalize to unit area for shape comparison
    area_d = np.sum(hd) * np.diff(edges)[0]
    area_g = np.sum(hg) * np.diff(edges)[0]
    hd_n, hd_err_n = hd / area_d, hd_err / area_d
    hg_n, hg_err_n = hg / area_g, hg_err / area_g

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1]})

    ax1.errorbar(centers, hd_n, yerr=hd_err_n, fmt="o", ms=3, color="k",
                 label="data (held-out)")
    ax1.stairs(hg_n, edges, color="tab:red", label="gen", lw=1.5)
    ax1.set_ylabel("normalized density")
    ax1.set_title(feat)
    ax1.legend()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hd_n > 0, hg_n / hd_n, np.nan)
        ratio_err = np.where(hd_n > 0, hg_err_n / hd_n, np.nan)
    ax2.errorbar(centers, ratio, yerr=ratio_err, fmt="o", ms=3, color="tab:red")
    ax2.axhline(1.0, color="k", lw=1, ls="--")
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel("gen / data")
    ax2.set_xlabel(feat)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"marginal_{feat}.png"), dpi=140)
    plt.close(fig)


# ==========================================================================
# 4. Joint / correlation structure
# ==========================================================================

def compare_correlations(data_events: pd.DataFrame, gen_events: pd.DataFrame,
                          weight_col: Optional[str], feature_list: List[str],
                          out_dir: Optional[str] = None) -> dict:
    """
    Weighted Pearson correlation matrices for data and gen, their
    difference, and a pairwise 2D-histogram corner plot -- this is where
    a model can pass every 1D check and still be wrong (independent
    sampling of otherwise-correlated features).
    """
    _safe_makedirs(out_dir)
    w_data = _get_weights(data_events, weight_col)
    w_gen = _get_weights(gen_events, weight_col)

    Xd = data_events[feature_list].to_numpy(dtype=float)
    Xg = gen_events[feature_list].to_numpy(dtype=float)

    corr_d = _weighted_corr_matrix(Xd, w_data)
    corr_g = _weighted_corr_matrix(Xg, w_gen)
    diff = corr_g - corr_d
    max_abs_diff = float(np.nanmax(np.abs(diff[~np.eye(len(feature_list), dtype=bool)])))

    if out_dir:
        _plot_corr_matrices(corr_d, corr_g, feature_list, out_dir)
        _plot_corner(Xd, w_data, Xg, w_gen, feature_list, out_dir)

    return {
        "corr_data": pd.DataFrame(corr_d, index=feature_list, columns=feature_list),
        "corr_gen": pd.DataFrame(corr_g, index=feature_list, columns=feature_list),
        "corr_diff": pd.DataFrame(diff, index=feature_list, columns=feature_list),
        "max_abs_corr_diff": max_abs_diff,
    }


def _weighted_corr_matrix(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    mean = np.sum(X * w[:, None], axis=0)
    Xc = X - mean
    cov = (Xc * w[:, None]).T @ Xc
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, np.nan)
    return corr


def _plot_corr_matrices(corr_d, corr_g, feature_list, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    diff = corr_g - corr_d
    for ax, mat, title, cmap, vlim in zip(
            axes, [corr_d, corr_g, diff],
            ["data (held-out)", "gen", "gen - data"],
            ["coolwarm", "coolwarm", "coolwarm"],
            [(-1, 1), (-1, 1), (-0.3, 0.3)]):
        im = ax.imshow(mat, vmin=vlim[0], vmax=vlim[1], cmap=cmap)
        ax.set_xticks(range(len(feature_list)))
        ax.set_yticks(range(len(feature_list)))
        ax.set_xticklabels(feature_list, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_list, fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_matrices.png"), dpi=140)
    plt.close(fig)


def _plot_corner(Xd, wd, Xg, wg, feature_list, out_dir, max_points=4000, bins=30):
    n = len(feature_list)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    rng = np.random.default_rng(0)

    def subsample(X, w, k):
        if len(X) <= k:
            return X, w
        p = w / w.sum()
        idx = rng.choice(len(X), size=k, replace=False, p=p)
        return X[idx], w[idx]

    Xd_s, wd_s = subsample(Xd, wd, max_points)
    Xg_s, wg_s = subsample(Xg, wg, max_points)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                lo = min(Xd[:, i].min(), Xg[:, i].min())
                hi = max(Xd[:, i].max(), Xg[:, i].max())
                edges = np.linspace(lo, hi, bins)
                ax.hist(Xd[:, i], bins=edges, weights=wd, density=True,
                        histtype="step", color="k", label="data")
                ax.hist(Xg[:, i], bins=edges, weights=wg, density=True,
                        histtype="step", color="tab:red", label="gen")
                if i == 0:
                    ax.legend(fontsize=6)
            elif i > j:
                ax.scatter(Xd_s[:, j], Xd_s[:, i], s=2, alpha=0.25, color="k",
                           label="data")
                ax.scatter(Xg_s[:, j], Xg_s[:, i], s=2, alpha=0.25,
                           color="tab:red", label="gen")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(feature_list[j], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(feature_list[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "corner_plot.png"), dpi=140)
    plt.close(fig)


# ==========================================================================
# 5. Classifier two-sample test
# ==========================================================================

def classifier_two_sample_test(data_events: pd.DataFrame, gen_events: pd.DataFrame,
                                weight_col: Optional[str], feature_list: List[str],
                                n_runs: int = 5, test_size: float = 0.3,
                                out_dir: Optional[str] = None) -> dict:
    """
    Train a HistGradientBoostingClassifier to discriminate gen from data
    (label 1 = data, 0 = gen), using the GEN importance weights so the
    classifier is judged in the same weighted sense the physics analysis
    will use. AUC ~ 0.5 means gen is statistically indistinguishable from
    data on these features (jointly, not just marginally).

    Also reports single-feature AUCs, which help localize *where* any
    mismodeling lives if the full-feature AUC is significantly above 0.5.
    """
    _safe_makedirs(out_dir)
    w_data = _get_weights(data_events, weight_col)
    w_gen = _get_weights(gen_events, weight_col)

    Xd = data_events[feature_list].to_numpy(dtype=float)
    Xg = gen_events[feature_list].to_numpy(dtype=float)
    X = np.concatenate([Xd, Xg], axis=0)
    y = np.concatenate([np.ones(len(Xd)), np.zeros(len(Xg))])
    w = np.concatenate([w_data, w_gen])

    full_aucs = []
    last_fpr, last_tpr = None, None
    for run in range(n_runs):
        X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
            X, y, w, test_size=test_size, random_state=run, stratify=y)
        clf = HistGradientBoostingClassifier(random_state=run, max_depth=4)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        p = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p, sample_weight=w_te)
        full_aucs.append(auc)
        if run == 0:
            last_fpr, last_tpr, _ = roc_curve(y_te, p, sample_weight=w_te)

    full_auc_mean, full_auc_std = float(np.mean(full_aucs)), float(np.std(full_aucs))

    # single-feature AUCs to localize mismodeling
    per_feature_auc = {}
    for k, feat in enumerate(feature_list):
        X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
            X[:, [k]], y, w, test_size=test_size, random_state=0, stratify=y)
        clf = HistGradientBoostingClassifier(random_state=0, max_depth=3)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        p = clf.predict_proba(X_te)[:, 1]
        per_feature_auc[feat] = float(roc_auc_score(y_te, p, sample_weight=w_te))

    if out_dir and last_fpr is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(last_fpr, last_tpr, color="tab:blue",
                label=f"gen vs. data (AUC={full_auc_mean:.3f}\u00b1{full_auc_std:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="random (AUC=0.5)")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("Classifier two-sample test ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "classifier_two_sample_roc.png"), dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        feats_sorted = sorted(per_feature_auc, key=per_feature_auc.get, reverse=True)
        ax.barh(feats_sorted, [per_feature_auc[f] for f in feats_sorted],
                color="tab:orange")
        ax.axvline(0.5, color="k", ls="--")
        ax.set_xlabel("single-feature AUC (gen vs. data)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_feature_auc.png"), dpi=140)
        plt.close(fig)

    return {
        "full_auc_mean": full_auc_mean,
        "full_auc_std": full_auc_std,
        "full_auc_runs": full_aucs,
        "per_feature_auc": per_feature_auc,
    }


# ==========================================================================
# 6. GEN importance-weight diagnostics
# ==========================================================================

def weight_diagnostics(gen_events: pd.DataFrame, weight_col: Optional[str],
                        extreme_sigma: float = 3.0,
                        out_dir: Optional[str] = None) -> dict:
    """
    Diagnose the w(m) reweighting used to build gen_events (see Sec. 2.2 of
    2311.12924). A degenerate weight distribution (small effective sample
    size, or a large fraction of the total weight sitting in a few extreme
    events) is a direct symptom of extrapolation failure, independent of
    anything a marginal/correlation check would show.
    """
    _safe_makedirs(out_dir)
    if weight_col is None or weight_col not in gen_events.columns:
        return {"note": f"No weight column '{weight_col}' found; "
                         f"assuming unweighted (all weights = 1)."}

    w = gen_events[weight_col].to_numpy(dtype=float)
    n = len(w)
    ess = _effective_sample_size(w)
    mean_w, std_w = w.mean(), w.std()
    cut = mean_w + extreme_sigma * std_w
    extreme_mask = w > cut
    frac_events_extreme = extreme_mask.mean()
    frac_weight_extreme = w[extreme_mask].sum() / w.sum() if w.sum() > 0 else np.nan

    if out_dir:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(w, bins=60, color="tab:purple", alpha=0.8)
        ax.axvline(cut, color="k", ls="--",
                   label=f"{extreme_sigma}\u03c3 above mean")
        ax.set_yscale("log")
        ax.set_xlabel(f"weight ({weight_col})")
        ax.set_ylabel("events")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "weight_distribution.png"), dpi=140)
        plt.close(fig)

    return {
        "n_events": n,
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / n if n else np.nan,
        "mean_weight": float(mean_w),
        "std_weight": float(std_w),
        "n_extreme_weight_events": int(extreme_mask.sum()),
        "frac_events_extreme": float(frac_events_extreme),
        "frac_total_weight_in_extreme_events": float(frac_weight_extreme),
    }


# ==========================================================================
# 7. Nearest-neighbour coverage / local-density check
# ==========================================================================

def nearest_neighbor_check(data_events: pd.DataFrame, gen_events: pd.DataFrame,
                            feature_list: List[str], weight_col: Optional[str] = None,
                            n_samples: int = 3000, random_state: int = 0,
                            out_dir: Optional[str] = None) -> dict:
    """
    Compares (a) the distribution of nearest-neighbour distances *within*
    the held-out data sample to (b) the distribution of nearest-neighbour
    distances from each gen event to its closest data event, after
    standardizing all features by the data mean/std.

    Gen events landing much closer to individual data points than data's
    own typical nearest-neighbour spacing can indicate the generator is
    collapsing onto a sparse set of modes (under-covering phase space);
    systematically larger gen-to-data distances indicate the generator is
    missing regions of support that real data occupies.
    """
    _safe_makedirs(out_dir)
    rng = np.random.default_rng(random_state)

    Xd = data_events[feature_list].to_numpy(dtype=float)
    Xg = gen_events[feature_list].to_numpy(dtype=float)

    mu, sigma = Xd.mean(axis=0), Xd.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xd_z = (Xd - mu) / sigma
    Xg_z = (Xg - mu) / sigma

    def subsample(X, k):
        if len(X) <= k:
            return X
        idx = rng.choice(len(X), size=k, replace=False)
        return X[idx]

    Xd_sub = subsample(Xd_z, n_samples)
    Xg_sub = subsample(Xg_z, n_samples)

    # (a) data's own leave-one-out NN distance
    nn_data = NearestNeighbors(n_neighbors=2).fit(Xd_z)
    dist_dd, _ = nn_data.kneighbors(Xd_sub)
    dist_dd = dist_dd[:, 1]  # skip self (distance 0)

    # (b) gen -> nearest data event
    dist_gd, _ = nn_data.kneighbors(Xg_sub, n_neighbors=1)
    dist_gd = dist_gd[:, 0]

    ks_stat, ks_p = stats.ks_2samp(dist_dd, dist_gd)

    if out_dir:
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(0, np.percentile(np.concatenate([dist_dd, dist_gd]), 99), 60)
        ax.hist(dist_dd, bins=bins, density=True, histtype="step", color="k",
                label="data \u2192 nearest data (self-spacing)")
        ax.hist(dist_gd, bins=bins, density=True, histtype="step", color="tab:red",
                label="gen \u2192 nearest data")
        ax.set_xlabel("standardized Euclidean distance")
        ax.set_ylabel("density")
        ax.set_title(f"NN-distance coverage check (KS p={ks_p:.3g})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "nearest_neighbor_check.png"), dpi=140)
        plt.close(fig)

    return {
        "median_data_self_spacing": float(np.median(dist_dd)),
        "median_gen_to_data_distance": float(np.median(dist_gd)),
        "ratio_median_gen_over_data": float(np.median(dist_gd) / np.median(dist_dd)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


# ==========================================================================
# Orchestrator
# ==========================================================================

def run_full_validation(
        data_events: pd.DataFrame,
        gen_events: pd.DataFrame,
        weight_col: Optional[str],
        feature_list: List[str],
        bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        out_dir: str = "gen_validation_plots",
        n_classifier_runs: int = 5,
        tail_quantile: float = 0.95,
        auc_warn_threshold: float = 0.55,
        auc_fail_threshold: float = 0.65,
        ks_p_threshold: float = 0.01,
        verbose: bool = True,
) -> dict:
    """
    Run every check above and return a single structured report. Also
    saves all diagnostic plots into `out_dir` and prints a short pass/
    warn/fail-style summary (thresholds are adjustable; they are
    heuristics, not hard cutoffs -- use the plots and numbers, not just
    the summary line, before concluding anything).
    """
    _safe_makedirs(out_dir)
    report: dict = {}

    report["bounds"] = check_bounds(gen_events, feature_list, bounds)
    report["moments"] = compare_moments(data_events, gen_events, weight_col, feature_list)
    report["marginals"] = compare_marginals(
        data_events, gen_events, weight_col, feature_list,
        tail_quantile=tail_quantile, out_dir=out_dir)
    report["correlations"] = compare_correlations(
        data_events, gen_events, weight_col, feature_list, out_dir=out_dir)
    report["classifier_two_sample"] = classifier_two_sample_test(
        data_events, gen_events, weight_col, feature_list,
        n_runs=n_classifier_runs, out_dir=out_dir)
    report["weights"] = weight_diagnostics(gen_events, weight_col, out_dir=out_dir)
    report["nearest_neighbor"] = nearest_neighbor_check(
        data_events, gen_events, feature_list, weight_col=weight_col, out_dir=out_dir)

    if verbose:
        _print_summary(report, feature_list, auc_warn_threshold,
                        auc_fail_threshold, ks_p_threshold)

    _save_json_report(report, os.path.join(out_dir, "validation_report.json"))
    return report


def _flag(condition_bad, condition_warn):
    if condition_bad:
        return "FAIL"
    if condition_warn:
        return "WARN"
    return "OK"


def _print_summary(report, feature_list, auc_warn, auc_fail, ks_p_thresh):
    print("=" * 70)
    print("GEN VALIDATION SUMMARY")
    print("=" * 70)

    # bounds
    bad_bounds = [f for f, d in report["bounds"].items()
                  if d["frac_below_lo"] > 0 or d["frac_above_hi"] > 0
                  or d["n_nan_or_inf"] > 0]
    print(f"\n[1] Physical bounds: "
          f"{'OK' if not bad_bounds else 'FAIL -> ' + ', '.join(bad_bounds)}")

    # marginals
    print("\n[2] Marginal shape (weighted KS test, per feature):")
    for _, row in report["marginals"].iterrows():
        flag = _flag(row["ks_pvalue"] < ks_p_thresh / 10, row["ks_pvalue"] < ks_p_thresh)
        print(f"    {row['feature']:>16s}: KS p={row['ks_pvalue']:.4f}  "
              f"Wasserstein(norm)={row['wasserstein_norm']:.4f}  "
              f"tail ratio(gen/data)={row['tail_ratio_gen_over_data']:.3f}  [{flag}]")

    # correlations
    max_diff = report["correlations"]["max_abs_corr_diff"]
    flag = _flag(max_diff > 0.15, max_diff > 0.05)
    print(f"\n[3] Correlations: max |corr_gen - corr_data)| = {max_diff:.3f}  [{flag}]")

    # classifier
    auc = report["classifier_two_sample"]["full_auc_mean"]
    auc_std = report["classifier_two_sample"]["full_auc_std"]
    flag = _flag(auc > auc_fail, auc > auc_warn)
    print(f"\n[4] Classifier two-sample test: AUC = {auc:.3f} +/- {auc_std:.3f} "
          f"(0.5 = indistinguishable)  [{flag}]")
    for feat, a in sorted(report["classifier_two_sample"]["per_feature_auc"].items(),
                           key=lambda kv: -kv[1]):
        print(f"      per-feature AUC  {feat:>16s}: {a:.3f}")

    # weights
    wdiag = report["weights"]
    if "effective_sample_fraction" in wdiag:
        esf = wdiag["effective_sample_fraction"]
        flag = _flag(esf < 0.3, esf < 0.6)
        print(f"\n[5] Weight diagnostics: effective sample size = "
              f"{esf * 100:.1f}% of nominal, "
              f"{wdiag['frac_total_weight_in_extreme_events'] * 100:.2f}% of total "
              f"weight in {wdiag['n_extreme_weight_events']} extreme events  [{flag}]")
    else:
        print(f"\n[5] Weight diagnostics: {wdiag.get('note')}")

    # nearest neighbor
    nn = report["nearest_neighbor"]
    ratio = nn["ratio_median_gen_over_data"]
    flag = _flag(ratio < 0.3 or ratio > 3.0, ratio < 0.5 or ratio > 2.0)
    print(f"\n[6] Nearest-neighbor coverage: median(gen->data) / "
          f"median(data self-spacing) = {ratio:.2f}  [{flag}]")

    print("\n" + "=" * 70)
    print(f"Plots and full numeric report saved to disk (validation_report.json).")
    print("=" * 70)


def _save_json_report(report, path):
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return json.loads(o.to_json(orient="records"))
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        return o

    with open(path, "w") as f:
        json.dump(convert(report), f, indent=2, default=str)