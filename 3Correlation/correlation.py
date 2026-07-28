#!/usr/bin/env python3
"""
gpu_util_tps_correlation.py
===========================

Correlate GPU utilization with throughput (tokens/second) for vLLM /
Transformers per-step roofline runs, and draw correlation heatmaps between
GPU utilization, (idle-subtracted) power, and tokens/second.

Works on the per-step CSVs produced by step_metric5.py (vLLM) and
step_metric_transformers.py (HF).  It reads columns *by name*, so it copes
with both schemas:

  * vLLM  model*.csv          -> 24 cols, no `drain_like` column
  * HF    *_hf_metrics.csv    -> 25 cols, includes a `drain_like` column

What it does
------------
1.  Loads any number of CSV files (files, globs, or directories).
2.  Splits rows into `prefill` and `decode` phases.
3.  Drops vLLM v1 "drain" decode steps (the ~1 ms terminal iteration whose
    fabricated tok/s would swamp the correlation).  Uses the CSV's own
    `drain_like` flag when present, otherwise re-derives it the same way the
    pipeline does: decode latency < 0.5 x median decode latency, per file.
4.  Prints Pearson, Spearman and Kendall (tau-b) correlations between GPU
    utilization and tok/s -- for prefill and for decode -- overall and per
    group (a group defaults to the file's parent-directory name, so pooling
    files from several run folders keeps them separable).
5.  Renders, per phase, a row of three annotated 3x3 correlation heatmaps
    (Pearson / Spearman / Kendall) among GPU util, power and tok/s.
    Also writes the matrices to CSV.
6.  Renders, per phase, a util-vs-tok/s SCATTER (one panel per group, points
    colored by model, with the three coefficients + a least-squares line) so
    you can read the point cloud, not just the number. `--pairs` adds a full
    3-variable scatter-matrix. Turn these off with --no-scatter / --no-heatmap.

Prefill and decode are ALWAYS treated as two separate populations -- the
script never pools prefill and decode rows into one correlation.  "Pooling"
only ever means combining multiple files/devices/engines *within* a single
phase.

The three methods
-----------------
  Pearson   -- linear correlation of the raw values.
  Spearman  -- Pearson on the ranks; monotonic, robust to non-linearity.
  Kendall   -- tau-b; based on concordant/discordant pairs, robust to
               outliers and small samples.  (This is the "one you forgot".)

Dependencies
------------
  Required : Python 3.7+, numpy
  Optional : scipy       -> exact coefficients + p-values (recommended)
             matplotlib  -> PNG heatmaps (falls back to CSV/text if absent)

The coefficients are implemented in pure numpy, so the script still runs and
prints every correlation on a bare venv that only has numpy.  scipy, when
present, is used for canonical coefficients and p-values; matplotlib is only
needed to draw the heatmap images.

Usage
-----
  # one run folder (all of model1..4 pooled as one group):
  python gpu_util_tps_correlation.py rooftop_final/jetson-a-code/model*.csv

  # a whole directory (expands to the CSVs inside it):
  python gpu_util_tps_correlation.py rooftop_final/jetson-b-tf

  # everything combined -> overall correlation + a per-folder breakdown:
  python gpu_util_tps_correlation.py \
      thor-vllm/model*.csv  thor-tf/*_hf_metrics.csv \
      a100-vllm/model*.csv  a100-tf/*_hf_metrics.csv \
      --outdir corr_out --out-prefix combined

  # keep the drain steps, or use raw (not idle-subtracted) power:
  python gpu_util_tps_correlation.py *.csv --keep-drain
  python gpu_util_tps_correlation.py *.csv --power-col gpu_power_w_avg

Run with -h for the full option list.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import sys

import numpy as np

# ---- optional dependencies -------------------------------------------------
try:
    import scipy.stats as _sps
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")                           # headless / cluster-safe
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:                                   # pragma: no cover
    _HAVE_MPL = False


# ===========================================================================
# Correlation primitives (pure numpy; scipy used when available)
# ===========================================================================
def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with average ranks for ties (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = a.argsort(kind="mergesort")
    sa = a[order]
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0            # average of the tie block
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = r
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b, with the same tie handling scipy uses."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 2:
        return float("nan")
    # numerator = sum_{i<j} sign(x_i - x_j) * sign(y_i - y_j) = C - D
    num = 0.0
    for i in range(n - 1):
        num += float(np.dot(np.sign(x[i] - x[i + 1:]),
                            np.sign(y[i] - y[i + 1:])))
    n0 = n * (n - 1) / 2.0

    def tie_pairs(a: np.ndarray) -> float:
        _, counts = np.unique(a, return_counts=True)
        return float(np.sum(counts * (counts - 1) / 2.0))

    n1 = tie_pairs(x)                               # pairs tied in x
    n2 = tie_pairs(y)                               # pairs tied in y
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return float("nan")
    return float(num / denom)


_METHODS = ("pearson", "spearman", "kendall")
_METHOD_LABEL = {"pearson": "Pearson", "spearman": "Spearman",
                 "kendall": "Kendall tau-b"}


def single_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Coefficient only, on finite-complete pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    if _HAVE_SCIPY:
        if method == "pearson":
            return float(_sps.pearsonr(x, y)[0])
        if method == "spearman":
            return float(_sps.spearmanr(x, y)[0])
        if method == "kendall":
            return float(_sps.kendalltau(x, y)[0])
    if method == "pearson":
        return _pearson(x, y)
    if method == "spearman":
        return _spearman(x, y)
    if method == "kendall":
        return _kendall_tau_b(x, y)
    raise ValueError(f"unknown method {method!r}")


def corr_with_p(x: np.ndarray, y: np.ndarray):
    """All three methods -> {method: (coef, p_or_None, n)} on complete pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    out = {}
    degenerate = n < 3 or x.std() == 0 or y.std() == 0
    for method in _METHODS:
        if degenerate:
            out[method] = (float("nan"), None, n)
        elif _HAVE_SCIPY:
            fn = {"pearson": _sps.pearsonr,
                  "spearman": _sps.spearmanr,
                  "kendall": _sps.kendalltau}[method]
            res = fn(x, y)
            out[method] = (float(res[0]), float(res[1]), n)
        else:
            out[method] = (single_corr(x, y, method), None, n)
    return out


# ===========================================================================
# Data loading
# ===========================================================================
def _to_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def expand_inputs(patterns, recursive=False):
    """Turn files / globs / directories into a de-duplicated CSV file list."""
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            sub = ("**/*.csv" if recursive else "*.csv")
            matched = glob.glob(os.path.join(pat, sub), recursive=recursive)
        elif any(ch in pat for ch in "*?[]"):
            matched = glob.glob(pat, recursive=recursive)
        else:
            matched = [pat]
        for f in matched:
            if f.lower().endswith(".csv") and os.path.isfile(f):
                files.append(f)
    # de-dupe, keep order
    seen, uniq = set(), []
    for f in files:
        ap = os.path.abspath(f)
        if ap not in seen:
            seen.add(ap)
            uniq.append(f)
    return uniq


def group_key(path, mode):
    if mode == "none":
        return "all"
    if mode == "file":
        return os.path.splitext(os.path.basename(path))[0]
    # default: parent directory name (encodes device+engine in Sean's layout)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.splitext(os.path.basename(path))[0]


# ===========================================================================
# Reporting
# ===========================================================================
VAR_KEYS = ("util", "power", "tps")


def var_labels(power_col):
    ptxt = "Dyn power" if power_col == "gpu_power_w_dyn" else "Power"
    return {"util": "GPU util", "power": ptxt, "tps": "Tok/s"}


def _fmt(coef, p):
    if coef is None or (isinstance(coef, float) and math.isnan(coef)):
        c = "   nan"
    else:
        c = f"{coef:+.3f}"
    if p is None:
        return f"{c}   {'':>9}"
    if isinstance(p, float) and math.isnan(p):
        return f"{c}   {'nan':>9}"
    return f"{c}   {p:9.2e}"


def print_util_vs_tps(rows, phases, title):
    print()
    print("=" * 74)
    print(f"  GPU utilization  vs  tokens/second        [{title}]")
    print("=" * 74)
    hdr_p = "p-value" if _HAVE_SCIPY else "p-value*"
    print(f"  {'phase':<9}{'method':<15}{'coef':>7}   {hdr_p:>9}   {'n':>6}")
    print("  " + "-" * 70)
    for phase in phases:
        sel = [r for r in rows if r["phase"] == phase]
        util = np.array([r["util"] for r in sel], float)
        tps = np.array([r["tps"] for r in sel], float)
        res = corr_with_p(util, tps)
        for method in _METHODS:
            coef, p, n = res[method]
            print(f"  {phase:<9}{_METHOD_LABEL[method]:<15}"
                  f"{_fmt(coef, p)}   {n:>6}")
        print("  " + "-" * 70)
    if not _HAVE_SCIPY:
        print("  * install scipy for p-values (coefficients shown are exact).")


def corr_matrix(rows, method, power_col):
    """3x3 correlation matrix among util/power/tps on complete rows."""
    cols = {k: np.array([r[k] for r in rows], float) for k in VAR_KEYS}
    finite = np.ones(len(rows), bool)
    for k in VAR_KEYS:
        finite &= np.isfinite(cols[k])
    for k in VAR_KEYS:
        cols[k] = cols[k][finite]
    n = int(finite.sum())
    M = np.eye(len(VAR_KEYS))
    for i, a in enumerate(VAR_KEYS):
        for j, b in enumerate(VAR_KEYS):
            if i < j:
                c = single_corr(cols[a], cols[b], method)
                M[i, j] = M[j, i] = c
    return M, n


def print_matrix(M, n, labels, method, phase):
    names = [labels[k] for k in VAR_KEYS]
    print()
    print(f"  [{phase}]  {_METHOD_LABEL[method]}  (n={n})")
    print("    " + "".join(f"{nm:>12}" for nm in names))
    for i, k in enumerate(VAR_KEYS):
        cells = "".join(
            ("      nan  " if math.isnan(M[i, j]) else f"{M[i, j]:>+11.3f}")
            for j in range(len(VAR_KEYS)))
        print(f"  {labels[k]:>10}{cells}")


def write_matrix_csv(path, M, n, labels, method, phase):
    names = [labels[k] for k in VAR_KEYS]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"{method}", f"phase={phase}", f"n={n}"])
        w.writerow([""] + names)
        for i, k in enumerate(VAR_KEYS):
            w.writerow([labels[k]] + [f"{M[i, j]:.6f}" for j in range(3)])


# ===========================================================================
# Heatmaps
# ===========================================================================
def draw_heatmaps(rows, phase, labels, power_col, outpath, group_note=""):
    """One figure per phase: three annotated 3x3 heatmaps side by side."""
    if not _HAVE_MPL:
        return None
    names = [labels[k] for k in VAR_KEYS]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    n_used = None
    for ax, method in zip(axes, _METHODS):
        M, n = corr_matrix(rows, method, power_col)
        n_used = n
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(_METHOD_LABEL[method], fontsize=11, pad=8)
        # annotate; text color flips on dark cells for legibility
        for i in range(3):
            for j in range(3):
                val = M[i, j]
                txt = "nan" if math.isnan(val) else f"{val:.2f}"
                shade = 0.0 if math.isnan(val) else abs(val)
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color=("white" if shade > 0.55 else "black"))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("correlation coefficient", fontsize=9)
    ptxt = "idle-subtracted power" if power_col == "gpu_power_w_dyn" else power_col
    sub = f"phase = {phase}   |   n = {n_used}   |   power = {ptxt}"
    if group_note:
        sub += f"   |   {group_note}"
    fig.suptitle(f"GPU util  /  power  /  tokens-per-second correlation\n{sub}",
                 fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.82, bottom=0.16, wspace=0.35)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ===========================================================================
# Scatter plots
# ===========================================================================
def _finite_xy(rows, xk, yk):
    x = np.array([r[xk] for r in rows], float)
    y = np.array([r[yk] for r in rows], float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _coef_note(x, y):
    """Multi-line 'Pearson/Spearman/Kendall + n' annotation for a scatter."""
    res = corr_with_p(x, y)
    lines = []
    for mth in _METHODS:
        c, _p, _n = res[mth]
        cc = ("n/a" if c is None or (isinstance(c, float) and math.isnan(c))
              else f"{c:+.2f}")
        lines.append(f"{_METHOD_LABEL[mth].split()[0]:<8} {cc}")
    lines.append(f"n = {int(np.size(x))}")
    return "\n".join(lines)


def draw_scatter(rows, phase, labels, outpath):
    """util (x) vs tok/s (y), one panel per group, points colored by model.

    This is the picture the correlation coefficient summarizes: read it, not
    just the number. A dashed line is the least-squares (Pearson) trend."""
    if not _HAVE_MPL:
        return None
    groups = sorted({r["group"] for r in rows})
    # One consistent color per source across ALL panels, so a single shared
    # legend is truthful even if one group is missing a run (e.g. OOM skip).
    all_sources = sorted({r["source"] for r in rows})
    _cmap = plt.get_cmap("tab20" if len(all_sources) > 10 else "tab10")
    scolor = {s: _cmap(i % _cmap.N) for i, s in enumerate(all_sources)}
    ng = len(groups)
    ncols = 1 if ng <= 1 else (2 if ng <= 4 else 3)
    nrows = max(1, math.ceil(ng / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.9 * nrows),
                             squeeze=False)
    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        gr = [r for r in rows if r["group"] == g]
        for s in sorted({r["source"] for r in gr}):
            x, y = _finite_xy([r for r in gr if r["source"] == s], "util", "tps")
            if x.size:
                ax.scatter(x, y, s=11, alpha=0.55, color=scolor[s],
                           edgecolors="none")
        x, y = _finite_xy(gr, "util", "tps")
        if x.size:
            ax.text(0.03, 0.97, _coef_note(x, y), transform=ax.transAxes,
                    va="top", ha="left", fontsize=8.5, family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
            if x.size >= 3 and x.std() > 0:
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, b * xs + a, "--", color="0.35", lw=1, alpha=0.8)
            ax.set_title(f"{g}   (util std {x.std():.1f}%)", fontsize=10)
        else:
            ax.set_title(g, fontsize=10)
        ax.set_xlabel(f"{labels['util']} (%)")
        ax.set_ylabel("tokens / s")
        ax.grid(alpha=0.25)
    for k in range(ng, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    # ONE shared legend below the panels (every panel uses the same colors)
    # instead of a duplicate legend covering the data inside each panel.
    bottom = 0.0
    if len(all_sources) > 1:
        ncol_leg = min(8, len(all_sources))
        nrow_leg = math.ceil(len(all_sources) / ncol_leg)
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=scolor[s],
                              markersize=6, label=s) for s in all_sources]
        fig.legend(handles=handles, loc="lower center", ncol=ncol_leg,
                   fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))
        bottom = 0.03 + 0.03 * nrow_leg
    fig.suptitle(f"GPU utilization  vs  tokens / second   —   phase = {phase}"
                 f"   (each panel = one group; color = model)", fontsize=12)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def draw_pairs_matrix(rows, phase, labels, power_col, outpath):
    """3x3 scatter-matrix of util/power/tok-s: hist on the diagonal, scatter in
    the lower triangle (colored by group), correlation text in the upper."""
    if not _HAVE_MPL:
        return None
    keys = VAR_KEYS
    names = [labels[k] for k in keys]
    cols = {k: np.array([r[k] for r in rows], float) for k in keys}
    finite = np.ones(len(rows), bool)
    for k in keys:
        finite &= np.isfinite(cols[k])
    for k in keys:
        cols[k] = cols[k][finite]
    grp = np.array([r["group"] for r in rows])[finite]
    n = int(finite.sum())
    if n < 2:
        return None
    groups = sorted(set(grp.tolist()))
    cmap = plt.get_cmap("tab10")
    gcolor = {g: cmap(i % 10) for i, g in enumerate(groups)}
    div = plt.get_cmap("RdBu_r")
    K = len(keys)
    fig, axes = plt.subplots(K, K, figsize=(3.3 * K, 3.3 * K), squeeze=False)
    for i in range(K):
        for j in range(K):
            ax = axes[i][j]
            if i == j:
                ax.hist(cols[keys[i]], bins=30, color="0.6")
                ax.set_yticks([])
            elif i > j:
                for g in groups:
                    m = grp == g
                    ax.scatter(cols[keys[j]][m], cols[keys[i]][m], s=6,
                               alpha=0.5, color=gcolor[g], edgecolors="none")
                ax.grid(alpha=0.2)
            else:
                r = single_corr(cols[keys[j]], cols[keys[i]], "pearson")
                rs = single_corr(cols[keys[j]], cols[keys[i]], "spearman")
                rk = single_corr(cols[keys[j]], cols[keys[i]], "kendall")
                shade = 0.5 if math.isnan(r) else (r + 1) / 2
                ax.set_facecolor(div(shade))
                txt = ("n/a" if math.isnan(r)
                       else f"P {r:+.2f}\nS {rs:+.2f}\nK {rk:+.2f}")
                ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=11,
                        transform=ax.transAxes,
                        color=("white" if not math.isnan(r) and abs(r) > 0.55
                               else "black"))
                ax.set_xticks([])
                ax.set_yticks([])
            if i == K - 1:
                ax.set_xlabel(names[j], fontsize=9)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=9)
    bottom = 0.0
    if len(groups) > 1:
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=gcolor[g],
                              label=g) for g in groups]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(groups), 4), fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.005))
        bottom = 0.06
    fig.suptitle(f"Correlation scatter-matrix: {' / '.join(names)}"
                 f"   —   phase = {phase}   (n={n}; upper = P/S/K)", fontsize=12)
    fig.tight_layout(rect=[0, bottom, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Correlate GPU util with tok/s and heatmap util/power/tok-s "
                    "for vLLM & Transformers per-step roofline CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="*",
                    help="CSV files, globs, or directories. Default: "
                         "model*.csv and *_hf_metrics.csv in the current dir.")
    ap.add_argument("--power-col", default="gpu_power_w_dyn",
                    help="power column to use as the 'power' variable.")
    ap.add_argument("--util-col", default="gpu_util_pct_avg",
                    help="(informational) GPU utilization column name.")
    ap.add_argument("--tps-col", default="tokens_per_second",
                    help="(informational) tokens/second column name.")
    ap.add_argument("--group-by", choices=("dir", "file", "none"), default="dir",
                    help="how to group rows for the per-group breakdown.")
    ap.add_argument("--group", default=None,
                    help="force one group label for every input file "
                         "(handy for a single per-machine run).")
    ap.add_argument("--phases", default="prefill,decode",
                    help="comma list from {prefill,decode}. The two phases are "
                         "ALWAYS analyzed separately; prefill and decode rows "
                         "are never pooled together.")
    ap.add_argument("--keep-drain", action="store_true",
                    help="keep vLLM v1 terminal drain decode steps "
                         "(dropped by default).")
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into directories when expanding inputs.")
    ap.add_argument("--outdir", default=".", help="directory for PNG/CSV output.")
    ap.add_argument("--out-prefix", default="corr",
                    help="filename prefix for the output PNGs and CSVs.")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip ALL figures (still prints + writes CSV matrices).")
    ap.add_argument("--no-heatmap", action="store_true",
                    help="skip the correlation heatmap figures.")
    ap.add_argument("--no-scatter", action="store_true",
                    help="skip the util-vs-tok/s scatter figures "
                         "(drawn by default).")
    ap.add_argument("--pairs", action="store_true",
                    help="also draw a 3-variable (util/power/tok-s) "
                         "scatter-matrix per phase.")
    ap.add_argument("--per-group-heatmaps", action="store_true",
                    help="also render one heatmap figure per group.")
    args = ap.parse_args(argv)

    patterns = args.inputs or ["model*.csv", "*_hf_metrics.csv"]
    files = expand_inputs(patterns, recursive=args.recursive)
    if not files:
        ap.error("no CSV files matched. Pass files/globs/dirs, e.g. "
                 "rooftop_final/jetson-a-code/model*.csv")

    os.makedirs(args.outdir, exist_ok=True)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    labels = var_labels(args.power_col)

    # ---- load ----
    # patch the loader's expected util/tps keys by reading through fixed names
    rows, counts = load_rows_named(files, args.power_col, args.util_col,
                                   args.tps_col, args.group_by,
                                   args.keep_drain, args.group)

    if not rows:
        print("No usable rows loaded. Inputs must be step_metric CSVs with "
              "'phase', 'tokens_per_second' and 'gpu_util_pct_avg' columns.")
        return 1

    # ---- provenance banner ----
    print("=" * 74)
    print("  Loaded files")
    print("=" * 74)
    for path, gkey, n_tot, n_drain in counts:
        tag = f"  ({n_drain} drain step(s) "
        tag += "kept)" if args.keep_drain else "dropped)"
        print(f"  [{gkey}] {path}")
        print(f"        {n_tot} rows{tag if n_drain else ''}")
    engines = "scipy" if _HAVE_SCIPY else "numpy-only (no p-values)"
    if not (_HAVE_MPL and not args.no_plot):
        fig_desc = "disabled" if args.no_plot else "matplotlib not installed"
    else:
        parts = ([] if args.no_heatmap else ["heatmap"])
        parts += ([] if args.no_scatter else ["scatter"])
        parts += (["pairs-matrix"] if args.pairs else [])
        fig_desc = ", ".join(parts) if parts else "none"
    print(f"\n  power variable : {args.power_col}")
    print(f"  correlations   : {engines}")
    print(f"  figures        : {fig_desc}")
    n_pref = sum(1 for r in rows if r["phase"] == "prefill")
    n_dec = sum(1 for r in rows if r["phase"] == "decode")
    print(f"  usable rows    : {n_pref} prefill, {n_dec} decode "
          f"(after drain handling)")

    # ---- headline: util vs tok/s, overall + per group ----
    print_util_vs_tps(rows, phases, "all files pooled (prefill/decode kept separate)")
    groups = sorted({r["group"] for r in rows})
    if len(groups) > 1:
        for g in groups:
            print_util_vs_tps([r for r in rows if r["group"] == g],
                              phases, f"group: {g}")

    # ---- 3-variable matrices (text + CSV) and heatmaps, per phase ----
    print()
    print("=" * 74)
    print("  Correlation matrices : GPU util / power / tok-s")
    print("=" * 74)
    for phase in phases:
        sel = [r for r in rows if r["phase"] == phase]
        for method in _METHODS:
            M, n = corr_matrix(sel, method, args.power_col)
            print_matrix(M, n, labels, method, phase)
            csv_path = os.path.join(
                args.outdir, f"{args.out_prefix}_matrix_{phase}_{method}.csv")
            write_matrix_csv(csv_path, M, n, labels, method, phase)

        if _HAVE_MPL and not args.no_plot:
            if not args.no_heatmap:
                png = os.path.join(args.outdir,
                                   f"{args.out_prefix}_heatmap_{phase}.png")
                if draw_heatmaps(sel, phase, labels, args.power_col, png):
                    print(f"\n  wrote heatmap     : {png}")
            if not args.no_scatter:
                spng = os.path.join(args.outdir,
                                    f"{args.out_prefix}_scatter_{phase}.png")
                if draw_scatter(sel, phase, labels, spng):
                    print(f"  wrote scatter     : {spng}")
            if args.pairs:
                ppng = os.path.join(args.outdir,
                                    f"{args.out_prefix}_pairs_{phase}.png")
                if draw_pairs_matrix(sel, phase, labels, args.power_col, ppng):
                    print(f"  wrote pairs-matrix: {ppng}")
        elif not _HAVE_MPL and not args.no_plot:
            print("\n  [note] matplotlib not installed -> no PNGs "
                  "(matrices written as CSV). `pip install matplotlib`.")

    # ---- optional per-group heatmaps ----
    if args.per_group_heatmaps and _HAVE_MPL and not args.no_plot:
        for g in groups:
            for phase in phases:
                sel = [r for r in rows
                       if r["group"] == g and r["phase"] == phase]
                if len(sel) < 3:
                    continue
                png = os.path.join(
                    args.outdir,
                    f"{args.out_prefix}_heatmap_{g}_{phase}.png")
                draw_heatmaps(sel, phase, labels, args.power_col, png,
                              group_note=f"group: {g}")

    print("\nDone.")
    return 0


def load_rows_named(files, power_col, util_col, tps_col, group_mode,
                    keep_drain, group_override):
    """Loader that maps arbitrary util/tps column names onto util/tps keys."""
    rows = []
    per_file_counts = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            need = ("phase", tps_col, util_col)
            missing = [c for c in need if c not in header]
            if missing:
                print(f"[warn] {path}: missing columns {missing}; skipped.")
                continue
            if power_col not in header:
                print(f"[warn] {path}: no '{power_col}' column; "
                      f"power set to NaN for these rows.")
            has_drain_col = "drain_like" in header
            frows = list(reader)

        gkey = group_override or group_key(path, group_mode)
        src = os.path.splitext(os.path.basename(path))[0]

        thr = None
        if not has_drain_col:
            dec_lat = [_to_float(r.get("latency_s")) for r in frows
                       if (r.get("phase") or "").strip().lower() == "decode"]
            dec_lat = [v for v in dec_lat if math.isfinite(v)]
            if len(dec_lat) >= 4:
                thr = 0.5 * statistics.median(dec_lat)

        n_total = n_drain = 0
        for r in frows:
            phase = (r.get("phase") or "").strip().lower()
            lat = _to_float(r.get("latency_s"))
            if has_drain_col:
                drain = str(r.get("drain_like", "")).strip().lower() in (
                    "true", "1", "yes")
            else:
                drain = (phase == "decode" and thr is not None
                         and math.isfinite(lat) and lat < thr)
            n_total += 1
            n_drain += int(drain)
            rows.append({
                "source": src, "group": gkey, "phase": phase, "drain": drain,
                "latency_s": lat,
                "util": _to_float(r.get(util_col)),
                "power": _to_float(r.get(power_col)),
                "tps": _to_float(r.get(tps_col)),
            })
        per_file_counts.append((path, gkey, n_total, n_drain))

    if not keep_drain:
        rows = [r for r in rows if not r["drain"]]
    return rows, per_file_counts


if __name__ == "__main__":
    sys.exit(main())