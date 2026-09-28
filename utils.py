'''
Utility functions for the stroke risk-stratification notebook.

All functions are threshold-aware and imbalance-aware: at ~5.8% stroke prevalence,
accuracy is uninformative, so everything here reports sensitivity, PR AUC, subgroup
breakdowns and per-1,000-patient impact counts instead.

Functions
----------
- positive_scores(model, X):
    Returns the predicted probability of the positive class (stroke) from a fitted estimator.

- auc_report(y_true, y_score, name="model", plot=True, label="stroke"):
    Prints and returns ROC AUC, PR AUC, outcome prevalence and PR AUC lift over the
    prevalence baseline. Optionally plots the ROC and precision-recall curves.

- wilson_ci(k, n, z=1.96):
    Returns a Wilson score interval for a binomial proportion. Preferred over the normal
    approximation here because subgroup counts are small and rates are near zero.

- confusion_at(y_true, scores, threshold):
    Returns (TP, FP, FN, TN) at one threshold, treating score >= threshold as "refer".

- operating_point(y_true, scores, threshold, name=None):
    Returns clinical metrics at one threshold (sensitivity, specificity, PPV) together with
    per-1,000-patient impact counts: patients flagged, strokes caught, strokes missed,
    false alerts and referrals per stroke caught.

- threshold_recall_floor(y_true, scores, min_recall=0.80):
    Highest threshold whose sensitivity still meets the recall floor. Data-driven, so it
    moves with the sample (see bootstrap_threshold).

- threshold_workload(scores, max_flag_rate=0.20):
    Threshold that flags at most max_flag_rate of patients, i.e. a clinic capacity cap.
    Needs no labels, only the score distribution.

- threshold_cost_based(cost_fn_to_fp=20.0):
    Bayes-optimal threshold 1 / (1 + C_FN/C_FP) for CALIBRATED probabilities. Depends only
    on the stated cost ratio, not on the sample, which is why it is the threshold locked
    in Section 9.1.

- threshold_youden(y_true, scores):
    Threshold maximising Youden's J (sensitivity + specificity - 1). Statistical reference
    only: it encodes no clinical cost, so it is reported but not used.

- bootstrap_threshold(y_true, scores, method, n_boot=1000, seed=42, **kw):
    Refits a threshold-selection method over bootstrap resamples to show how stable the
    chosen threshold is. Used to justify locking the cost-based threshold over the
    recall-floor one.

- decision_curve(y_true, scores, thresholds):
    Net benefit (Vickers & Elkin, 2006) of the model against "refer everyone" and
    "refer no one" across a range of threshold probabilities.

- subgroup_report(X, y_true, scores, threshold, group_cols, min_pos=10):
    Sensitivity, false-positive rate, PPV and flag rate per subgroup at a fixed threshold,
    each with a Wilson CI. Subgroups with fewer than min_pos strokes are labelled
    "not evaluable" rather than dropped, so under-representation stays visible.

- subgroup_discrimination(X, y_true, scores, group_cols, min_pos=10):
    ROC AUC and PR AUC per subgroup (threshold-free). Used to test whether the model ranks
    patients as well inside an age band as it does across age bands.

- test_summary(y_true, scores, threshold):
    One-table final test report: ROC AUC, PR AUC, Brier score, and sensitivity,
    specificity, PPV, NPV and flag rate with Wilson CIs.

- tidy_global_effects(causal_result, per_unit=None):
    Converts an RAI CausalResult's global effects into percentage points of absolute stroke
    risk, rescaled to clinically meaningful units (e.g. glucose per +50 mg/dL).

- plot_effects(tbl, title=...):
    Forest plot of estimated causal effects with 95% CIs; effects whose interval excludes
    zero are highlighted.

- ThresholdedModel(model, threshold):
    Wrapper whose .predict() applies the locked clinical threshold instead of sklearn's
    default 0.5. Required by the RAI dashboard: at ~6% prevalence no patient scores above
    0.5, so without it every fairness and error view would see "no stroke" for everyone.

- ThresholdCentredModel(model, threshold):
    Counterfactuals only. DiCE hard-codes 0.5 as the decision boundary, so this shifts the
    probability on the log-odds scale to put the clinical threshold exactly at 0.5.
    Decisions and ranking are unchanged; the displayed probabilities are shifted.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve,
)

__all__ = [
    "positive_scores", "auc_report", "wilson_ci", "confusion_at", "operating_point",
    "threshold_recall_floor", "threshold_workload", "threshold_cost_based", "threshold_youden",
    "bootstrap_threshold", "decision_curve", "subgroup_report", "subgroup_discrimination",
    "test_summary", "tidy_global_effects", "plot_effects",
    "ThresholdedModel", "ThresholdCentredModel",
]


# ── scoring ────────────────────────────────────────────────────────────────────
def positive_scores(model, X):
    """Probability of the positive class (stroke)."""
    return model.predict_proba(X)[:, 1]

def auc_report(y_true, y_score, name="model", plot=True, label="stroke"):
    """ROC AUC and PR AUC with the prevalence baseline, plus ROC / PR curves."""
    prev = float(np.mean(y_true))
    roc = roc_auc_score(y_true, y_score); pr = average_precision_score(y_true, y_score)
    print(name); print(f"PR AUC: {pr:.3f}"); print(f"ROC AUC: {roc:.3f}")
    print(f"{label.capitalize()} prevalence: {prev:.3f}")
    print(f"PR AUC relative to prevalence baseline: {pr/prev:.2f} times")
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        fpr, tpr, _ = roc_curve(y_true, y_score); ax[0].plot(fpr, tpr, color="#4C72B0")
        ax[0].plot([0, 1], [0, 1], ls="--", color="grey"); ax[0].set_title(f"ROC — {name} (AUC {roc:.3f})")
        ax[0].set_xlabel("False-positive rate"); ax[0].set_ylabel("Sensitivity")
        p, r, _ = precision_recall_curve(y_true, y_score); ax[1].plot(r, p, color="#C44E52")
        ax[1].axhline(prev, ls="--", color="grey", label="prevalence"); ax[1].legend()
        ax[1].set_title(f"Precision–recall — {name} (AP {pr:.3f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
        plt.tight_layout(); plt.show()
    return {"name": name, "roc_auc": roc, "pr_auc": pr, "prevalence": prev}

# ── counts and clinical impact at a threshold ─────────────────────────────────
def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval for a proportion — stable for small n and rates near zero."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def confusion_at(y_true, scores, threshold):
    """(TP, FP, FN, TN) at one threshold, where score >= threshold means 'refer'."""
    y = np.asarray(y_true).astype(int); flag = np.asarray(scores) >= threshold
    return (int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()),
            int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum()))

def operating_point(y_true, scores, threshold, name=None):
    """Clinical metrics and per-1,000-patient impact at one threshold."""
    tp, fp, fn, tn = confusion_at(y_true, scores, threshold)
    n = tp + fp + fn + tn; flagged = tp + fp
    return {"Policy": name, "Threshold": threshold,
            "Sensitivity (recall)": tp / (tp + fn) if tp + fn else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "PPV": tp / flagged if flagged else np.nan,
            "Flagged per 1,000": 1000 * flagged / n, "Strokes caught per 1,000": 1000 * tp / n,
            "Strokes missed per 1,000": 1000 * fn / n, "False alerts per 1,000": 1000 * fp / n,
            "Referrals per stroke caught": flagged / tp if tp else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

# ── threshold-selection methods (fit on VALIDATION scores only) ────────────────
def threshold_recall_floor(y_true, scores, min_recall=0.80):
    """Highest threshold whose sensitivity is still >= min_recall."""
    y = np.asarray(y_true).astype(int); s = np.asarray(scores)
    for t in np.unique(s)[::-1]:
        tp, _, fn, _ = confusion_at(y, s, t)
        if tp / (tp + fn) >= min_recall:
            return float(t)
    return float(np.min(s))

def threshold_workload(scores, max_flag_rate=0.20):
    """Threshold that flags at most max_flag_rate of patients (capacity constraint)."""
    s = np.sort(np.asarray(scores))[::-1]; k = int(np.floor(max_flag_rate * len(s)))
    return float(s[k - 1]) if k > 0 else float(s[0] + 1e-9)

def threshold_cost_based(cost_fn_to_fp=20.0):
    """For CALIBRATED probabilities: refer when p > 1 / (1 + C_FN / C_FP)."""
    return 1.0 / (1.0 + cost_fn_to_fp)

def threshold_youden(y_true, scores):
    """Threshold maximising Youden's J. Statistical reference only — it encodes no clinical cost."""
    fpr, tpr, thr = roc_curve(y_true, scores); j = tpr - fpr
    return float(thr[np.argmax(j[1:]) + 1])

def bootstrap_threshold(y_true, scores, method, n_boot=1000, seed=42, **kw):
    """Re-runs a threshold method over bootstrap resamples to show how stable the threshold is."""
    rng = np.random.default_rng(seed); y = np.asarray(y_true); s = np.asarray(scores); out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() > 0:
            out.append(method(y[idx], s[idx], **kw))
    return np.array(out)

def decision_curve(y_true, scores, thresholds):
    """Net benefit (Vickers & Elkin 2006) of the model vs refer-all / refer-none."""
    y = np.asarray(y_true); n = len(y); prev = y.mean(); rows = []
    for t in thresholds:
        tp, fp, _, _ = confusion_at(y, scores, t)
        rows.append({"threshold": t, "Model": tp / n - fp / n * t / (1 - t),
                     "Refer everyone": prev - (1 - prev) * t / (1 - t), "Refer no one": 0.0})
    return pd.DataFrame(rows)

# ── subgroup reports ──────────────────────────────────────────────────────────
def subgroup_report(X, y_true, scores, threshold, group_cols, min_pos=10):
    """Sensitivity / FPR / PPV / flag rate per subgroup at a fixed threshold, with Wilson CIs."""
    y = pd.Series(np.asarray(y_true), index=X.index); s = pd.Series(np.asarray(scores), index=X.index)
    rows = []
    for col in group_cols:
        for val, idx in X.groupby(col, observed=True).groups.items():
            tp, fp, fn, tn = confusion_at(y.loc[idx].values, s.loc[idx].values, threshold)
            pos, neg = tp + fn, fp + tn
            lo, hi = wilson_ci(tp, pos) if pos else (np.nan, np.nan)
            rows.append({"Attribute": col, "Group": str(val), "n": len(idx), "Strokes": pos,
                         "Flag rate": (tp + fp) / len(idx),
                         "Sensitivity": tp / pos if pos else np.nan,
                         "Sens. 95% CI": f"{lo:.2f}–{hi:.2f}" if pos else "—",
                         "False-positive rate": fp / neg if neg else np.nan,
                         "PPV": tp / (tp + fp) if tp + fp else np.nan, "Missed strokes": fn,
                         "Evaluable?": "yes" if pos >= min_pos else f"no (<{min_pos} strokes)"})
    return pd.DataFrame(rows)

def subgroup_discrimination(X, y_true, scores, group_cols, min_pos=10):
    """ROC AUC / PR AUC per subgroup (threshold-free)."""
    y = np.asarray(y_true); s = np.asarray(scores); rows = []
    for col in group_cols:
        for val in pd.unique(X[col].dropna()):
            m = (X[col] == val).values; k = int(y[m].sum())
            ok = 0 < k < m.sum()
            rows.append({"Attribute": col, "Group": str(val), "n": int(m.sum()), "Strokes": k,
                         "Stroke rate": y[m].mean(),
                         "ROC AUC": roc_auc_score(y[m], s[m]) if ok else np.nan,
                         "PR AUC": average_precision_score(y[m], s[m]) if k else np.nan,
                         "Evaluable?": "yes" if k >= min_pos else f"no (<{min_pos} strokes)"})
    return pd.DataFrame(rows).sort_values(["Attribute", "Group"])

def test_summary(y_true, scores, threshold):
    """Final test-set report: discrimination, calibration and operating-point metrics with Wilson CIs."""
    tp, fp, fn, tn = confusion_at(y_true, scores, threshold)
    def fmt(k, n):
        lo, hi = wilson_ci(k, n); return f"{k/n:.3f} ({lo:.3f}–{hi:.3f})"
    return pd.DataFrame({"Metric": ["ROC AUC", "PR AUC (average precision)", "Brier score",
                                    "Sensitivity (95% CI)", "Specificity (95% CI)", "PPV (95% CI)",
                                    "NPV (95% CI)", "Flag rate"],
                         "Test value": [f"{roc_auc_score(y_true, scores):.3f}",
                                        f"{average_precision_score(y_true, scores):.3f}",
                                        f"{brier_score_loss(y_true, scores):.4f}",
                                        fmt(tp, tp + fn), fmt(tn, tn + fp), fmt(tp, tp + fp),
                                        fmt(tn, tn + fn), f"{(tp + fp) / (tp + fp + fn + tn):.3f}"]})

# ── causal-analysis tables ────────────────────────────────────────────────────
def tidy_global_effects(causal_result, per_unit=None):
    """RAI CausalResult.global_effects → effects in percentage points, rescaled per clinical unit."""
    per_unit = per_unit or {}; rows = []
    for _, r in causal_result.global_effects.reset_index().iterrows():
        is_num = r["feature_value"] == "num"; mult = per_unit.get(r["feature"], 1) if is_num else 1
        if is_num:
            contrast = f"per +{mult}"
        elif r["feature_value"] == "1v0":
            contrast = "yes vs no"
        else:
            treat, ctrl = r["feature_value"].rsplit("v", 1); contrast = f"{treat} vs {ctrl}"
        rows.append({"Treatment feature": r["feature"], "Contrast": contrast,
                     "Effect on stroke risk (pp)": 100 * r["point"] * mult,
                     "95% CI low (pp)": 100 * r["ci_lower"] * mult,
                     "95% CI high (pp)": 100 * r["ci_upper"] * mult, "p-value": r["p_value"]})
    return pd.DataFrame(rows)

def plot_effects(tbl, title="Estimated average causal effect on stroke risk"):
    """Forest plot of causal effects with 95% CIs; intervals excluding zero are highlighted."""
    t = tbl.iloc[::-1].reset_index(drop=True)
    labels = t["Treatment feature"] + "  (" + t["Contrast"] + ")"
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2))
    sig = ((t["95% CI low (pp)"] > 0) | (t["95% CI high (pp)"] < 0)).values
    for i, r in t.iterrows():
        c = "#C44E52" if sig[i] else "#8C8C8C"
        ax.errorbar(r["Effect on stroke risk (pp)"], i,
                    xerr=[[r["Effect on stroke risk (pp)"] - r["95% CI low (pp)"]],
                          [r["95% CI high (pp)"] - r["Effect on stroke risk (pp)"]]],
                    fmt="o", color=c, ecolor=c, capsize=3)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(t))); ax.set_yticklabels(labels)
    ax.set_xlabel("Change in absolute stroke risk (percentage points), 95% CI"); ax.set_title(title)
    plt.tight_layout(); return fig

# ── model wrappers for the RAI Toolbox ────────────────────────────────────────
class ThresholdedModel:
    """.predict() applies the locked clinical threshold instead of 0.5.
    Without it the dashboard predicts 'no stroke' for everyone (max risk < 0.5 at 6% prevalence)."""
    def __init__(self, model, threshold):
        self.model, self.threshold = model, float(threshold); self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def predict(self, X):
        return (self.model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

class ThresholdCentredModel(ThresholdedModel):
    """For counterfactuals only: DiCE treats 0.5 as the class boundary, so the probability is shifted on the
    log-odds scale so that the clinical threshold sits exactly at 0.5. Ranking and decisions are unchanged."""
    def predict_proba(self, X):
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)) - np.log(self.threshold / (1 - self.threshold))
        q = 1 / (1 + np.exp(-z)); return np.c_[1 - q, q]