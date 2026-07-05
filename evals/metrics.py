"""
metrics.py — Statistical metrics for evaluating eval reliability and significance.

Why this file exists:
  Raw eval scores (pass rate, avg score) are meaningless without statistical
  context. This module answers the questions that matter:
    - Is this eval consistent across runs? (Spearman)
    - Is this improvement real or noise? (Cohen's d, paired t-test)
    - Does the judge agree with humans? (Cohen's kappa)
    - How confident are we in the average? (Bootstrap CI)
    - Is the router well-calibrated? (Calibration data)

  All functions are pure math — zero LLM calls, zero API cost, instant.

Dependencies:
  - numpy (array operations)
  - scipy (statistical functions)
  Both are already in requirements.txt.
"""

import numpy as np
from scipy import stats


# ── SPEARMAN CORRELATION ──────────────────────────────────────────────────────

def spearman_correlation(scores_a: list, scores_b: list) -> dict:
    """
    Measure rank agreement between two eval runs.

    Use case: You ran the same test suite twice (or with two different judges).
    Do they rank the test cases in the same order?

    Why Spearman over Pearson?
      Spearman uses RANKS, not raw values. This means it's robust to
      different scoring scales (e.g., one judge uses 1-5, another uses 1-10)
      and isn't thrown off by outliers.

    Args:
        scores_a: Scores from run A (one per test case)
        scores_b: Scores from run B (one per test case, same order)

    Returns:
        Dict with:
          - rho:            Correlation coefficient (-1.0 to 1.0)
          - p_value:        Statistical significance
          - interpretation: Human-readable summary
    """
    if len(scores_a) != len(scores_b):
        return {
            "rho": 0.0,
            "p_value": 1.0,
            "interpretation": "ERROR: Arrays must have equal length.",
        }

    if len(scores_a) < 3:
        return {
            "rho": 0.0,
            "p_value": 1.0,
            "interpretation": "Need at least 3 data points for Spearman.",
        }

    rho, p_value = stats.spearmanr(scores_a, scores_b)

    # Handle NaN (happens when all values are identical)
    if np.isnan(rho):
        rho = 1.0  # Identical arrays = perfect correlation
        p_value = 0.0

    # Interpret
    abs_rho = abs(rho)
    if abs_rho >= 0.9:
        strength = "Very strong"
    elif abs_rho >= 0.7:
        strength = "Strong"
    elif abs_rho >= 0.5:
        strength = "Moderate"
    elif abs_rho >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak / no"

    sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
    interpretation = f"{strength} correlation (rho={rho:.3f}, p={p_value:.4f}, {sig})."

    return {
        "rho": round(float(rho), 4),
        "p_value": round(float(p_value), 4),
        "interpretation": interpretation,
    }


# ── COHEN'S d (EFFECT SIZE) ──────────────────────────────────────────────────

def cohens_d(current_scores: list, baseline_scores: list) -> dict:
    """
    Measure the practical significance of a score difference.

    Use case: You changed the model and scores went from 3.8 to 4.1.
    Is that a meaningful improvement or just noise?

    Why not just compare averages?
      A 0.3 difference means nothing without context. If the standard
      deviation is 2.0, that 0.3 is trivial. If the SD is 0.1, it's huge.
      Cohen's d normalizes the difference by the pooled standard deviation.

    Args:
        current_scores:  Scores from the new/current run
        baseline_scores: Scores from the baseline/previous run

    Returns:
        Dict with:
          - d:              Effect size (positive = improvement)
          - interpretation: Human-readable summary
    """
    if len(current_scores) < 2 or len(baseline_scores) < 2:
        return {
            "d": 0.0,
            "interpretation": "Need at least 2 data points in each group.",
        }

    current = np.array(current_scores, dtype=float)
    baseline = np.array(baseline_scores, dtype=float)

    n1, n2 = len(current), len(baseline)
    mean_diff = np.mean(current) - np.mean(baseline)

    # Pooled standard deviation
    var1 = np.var(current, ddof=1)
    var2 = np.var(baseline, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        d = 0.0 if mean_diff == 0 else float('inf')
    else:
        d = mean_diff / pooled_std

    # Interpret (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        size = "Negligible"
    elif abs_d < 0.5:
        size = "Small"
    elif abs_d < 0.8:
        size = "Medium"
    else:
        size = "Large"

    direction = "improvement" if d > 0 else "regression" if d < 0 else "no change"
    interpretation = f"{size} effect size (d={d:.3f}). {direction.capitalize()} vs baseline."

    return {
        "d": round(float(d), 4),
        "interpretation": interpretation,
    }


# ── PAIRED T-TEST ─────────────────────────────────────────────────────────────

def paired_ttest(current_scores: list, baseline_scores: list) -> dict:
    """
    Test whether the difference between two runs is statistically significant.

    Use case: Paired with Cohen's d. Cohen's d tells you HOW BIG the
    difference is; the t-test tells you if it's REAL (not random chance).

    Why paired?
      We're comparing scores on the SAME test cases across two runs.
      Pairing controls for case difficulty — a hard case is hard in both runs.

    Args:
        current_scores:  Scores from the new run (one per test case)
        baseline_scores: Scores from the baseline run (same cases, same order)

    Returns:
        Dict with:
          - t_stat:         Test statistic
          - p_value:        Two-tailed p-value
          - interpretation: Human-readable summary
    """
    if len(current_scores) != len(baseline_scores):
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "interpretation": "ERROR: Arrays must have equal length for paired test.",
        }

    if len(current_scores) < 3:
        return {
            "t_stat": 0.0,
            "p_value": 1.0,
            "interpretation": "Need at least 3 paired observations.",
        }

    t_stat, p_value = stats.ttest_rel(current_scores, baseline_scores)

    # Handle NaN (happens when all differences are zero)
    if np.isnan(t_stat):
        t_stat = 0.0
        p_value = 1.0

    if p_value < 0.01:
        sig = "Highly significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "Significant (p < 0.05)"
    elif p_value < 0.10:
        sig = "Marginally significant (p < 0.10)"
    else:
        sig = "Not significant (p >= 0.10)"

    interpretation = f"{sig}. t={t_stat:.3f}, p={p_value:.4f}."

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "interpretation": interpretation,
    }


# ── COHEN'S KAPPA (INTER-RATER RELIABILITY) ──────────────────────────────────

def cohens_kappa(judge_labels: list, human_labels: list) -> dict:
    """
    Measure agreement between the LLM judge and a human, beyond chance.

    Use case: A human labeled 20 test cases as PASS/FAIL. The LLM judge
    also labeled them. Do they agree beyond what random luck would produce?

    Why not just use accuracy?
      If 90% of cases are PASS, a judge that always says PASS gets 90%
      "accuracy" — but it's useless. Kappa corrects for this by subtracting
      the expected agreement due to chance.

    Args:
        judge_labels: List of labels from the judge (e.g., ["PASS", "FAIL", ...])
        human_labels: List of labels from a human (same order)

    Returns:
        Dict with:
          - kappa:          Agreement coefficient (-1.0 to 1.0)
          - interpretation: Human-readable summary
    """
    if len(judge_labels) != len(human_labels):
        return {
            "kappa": 0.0,
            "interpretation": "ERROR: Arrays must have equal length.",
        }

    if len(judge_labels) < 2:
        return {
            "kappa": 0.0,
            "interpretation": "Need at least 2 observations.",
        }

    # Get unique labels
    labels = list(set(judge_labels + human_labels))

    if len(labels) < 2:
        return {
            "kappa": 1.0,
            "interpretation": "Perfect agreement (only one label present).",
        }

    # Build confusion matrix manually (to avoid sklearn dependency)
    n = len(judge_labels)
    # Observed agreement
    observed_agree = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    p_observed = observed_agree / n

    # Expected agreement by chance
    p_expected = 0.0
    for label in labels:
        p_judge = sum(1 for j in judge_labels if j == label) / n
        p_human = sum(1 for h in human_labels if h == label) / n
        p_expected += p_judge * p_human

    # Kappa
    if p_expected == 1.0:
        kappa = 1.0  # Edge case: everyone agrees on everything
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)

    # Interpret (Landis & Koch conventions)
    if kappa > 0.80:
        strength = "Almost perfect"
    elif kappa > 0.60:
        strength = "Substantial"
    elif kappa > 0.40:
        strength = "Moderate"
    elif kappa > 0.20:
        strength = "Fair"
    elif kappa > 0.0:
        strength = "Slight"
    else:
        strength = "Poor / no"

    interpretation = f"{strength} agreement (kappa={kappa:.3f})."

    return {
        "kappa": round(float(kappa), 4),
        "interpretation": interpretation,
    }


# ── WEIGHTED COHEN'S KAPPA (ORDINAL SCORES) ───────────────────────────────────

def weighted_cohens_kappa(judge_scores: list, human_scores: list, max_score: int = 5) -> dict:
    """
    Measure agreement between LLM judge and human for ORDINAL scores (1-5).
    
    Use case: Human gives scores 1-5, LLM judge gives scores 1-5.
    Weighted kappa treats a difference of 1 point as less serious than
    a difference of &points.
    
    Args:
        judge_scores: List of integer scores from judge (1 to max_score)
        human_scores: List of integer scores from human (same order)
        max_score:    Maximum possible score (default 5)
        
    Returns:
        Dict with:
          - kappa:          Weighted agreement coefficient (-1.0 to 1.0)
          - linear_kappa:   Linear weighted kappa
          - quadratic_kappa: Quadratic weighted kappa (recommended for ordinal)
          - interpretation: Human-readable summary
    """
    if len(judge_scores) != len(human_scores):
        return {
            "kappa": 0.0,
            "linear_kappa": 0.0,
            "quadratic_kappa": 0.0,
            "interpretation": "ERROR: Arrays must have equal length.",
        }
    
    if len(judge_scores) < 2:
        return {
            "kappa": 0.0,
            "linear_kappa": 0.0,
            "quadratic_kappa": 0.0,
            "interpretation": "Need at least 2 observations.",
        }
    
    n = len(judge_scores)
    labels = list(range(1, max_score + 1))
    
    # Build observed agreement matrix
    observed = np.zeros((max_score, max_score), dtype=int)
    for j, h in zip(judge_scores, human_scores):
        if 1 <= j <= max_score and 1 <= h <= max_score:
            observed[j-1, h-1] += 1
    
    # Calculate observed proportions
    p_observed = observed / n
    
    # Calculate expected proportions (chance agreement)
    judge_marginals = np.sum(observed, axis=1) / n
    human_marginals = np.sum(observed, axis=0) / n
    p_expected = np.outer(judge_marginals, human_marginals)
    
    # Weight matrices
    linear_weights = np.zeros((max_score, max_score))
    quadratic_weights = np.zeros((max_score, max_score))
    
    for i in range(max_score):
        for j in range(max_score):
            diff = abs(i - j)
            linear_weights[i, j] = 1 - (diff / (max_score - 1))
            quadratic_weights[i, j] = 1 - (diff ** 2 / ((max_score - 1) ** 2))
    
    # Calculate weighted kappas
    # Linear weighted
    observed_weighted_linear = np.sum(p_observed * linear_weights)
    expected_weighted_linear = np.sum(p_expected * linear_weights)
    if expected_weighted_linear == 1.0:
        kappa_linear = 1.0
    else:
        kappa_linear = (observed_weighted_linear - expected_weighted_linear) / (1 - expected_weighted_linear)
    
    # Quadratic weighted (recommended for ordinal data)
    observed_weighted_quad = np.sum(p_observed * quadratic_weights)
    expected_weighted_quad = np.sum(p_expected * quadratic_weights)
    if expected_weighted_quad == 1.0:
        kappa_quad = 1.0
    else:
        kappa_quad = (observed_weighted_quad - expected_weighted_quad) / (1 - expected_weighted_quad)
    
    # Simple (unweighted) kappa for comparison
    observed_diag = np.sum(np.diag(p_observed))
    expected_diag = np.sum(np.diag(p_expected))
    if expected_diag == 1.0:
        kappa_simple = 1.0
    else:
        kappa_simple = (observed_diag - expected_diag) / (1 - expected_diag)
    
    # Interpretation
    kappa_value = kappa_quad  # Use quadratic as the primary metric
    if kappa_value < 0:
        interpretation = f"Poor agreement beyond chance (weighted κ={kappa_value:.3f})."
    elif kappa_value < 0.2:
        interpretation = f"Slight agreement (weighted κ={kappa_value:.3f})."
    elif kappa_value < 0.4:
        interpretation = f"Fair agreement (weighted κ={kappa_value:.3f})."
    elif kappa_value < 0.6:
        interpretation = f"Moderate agreement (weighted κ={kappa_value:.3f})."
    elif kappa_value < 0.8:
        interpretation = f"Substantial agreement (weighted κ={kappa_value:.3f})."
    else:
        interpretation = f"Almost perfect agreement (weighted κ={kappa_value:.3f})."
    
    return {
        "kappa": round(float(kappa_simple), 4),
        "linear_kappa": round(float(kappa_linear), 4),
        "quadratic_kappa": round(float(kappa_quad), 4),
        "interpretation": interpretation,
        "n": n,
        "max_score": max_score,
    }


# ── BOOTSTRAP CONFIDENCE INTERVAL ────────────────────────────────────────────

def bootstrap_ci(scores: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = None) -> dict:
    """
    Estimate the confidence interval for the mean score using bootstrapping.
    
    Use case: Your avg score is 4.1 — but how precise is that number?
    Bootstrapping resamples your data N times to estimate the uncertainty.
    
    Why bootstrap instead of normal CI?
      Normal CI assumes Gaussian distribution. Eval scores are often
      skewed (lots of 4s and 5s, few 1s). Bootstrap makes no distributional
      assumptions — it works on any shape.
    
    Args:
        scores:      List of scores to analyze
        n_bootstrap: Number of bootstrap resamples (1000 is standard)
        ci:          Confidence level (0.95 = 95% CI)
        seed:        Optional random seed for reproducibility (None = random)
        
    Returns:
        Dict with:
          - lower:  Lower bound of the CI
          - mean:   Point estimate (actual mean)
          - upper:  Upper bound of the CI
          - width:  CI width (upper - lower; smaller = more precise)
          - n:      Sample size
          - se:     Bootstrap standard error
          - method: "percentile" or "normal_approximation"
    """
    if not scores or len(scores) < 2:
        mean_val = float(scores[0]) if scores else 0.0
        return {
            "lower": mean_val,
            "mean": mean_val,
            "upper": mean_val,
            "width": 0.0,
            "n": len(scores) if scores else 0,
            "se": 0.0,
            "method": "insufficient_data",
        }
    
    scores_arr = np.array(scores, dtype=float)
    actual_mean = float(np.mean(scores_arr))
    n = len(scores_arr)
    
    # Use random seed if provided, otherwise truly random
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Resample with replacement and compute means
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute bootstrap standard error
    se = float(np.std(bootstrap_means, ddof=1))
    
    # Compute percentiles for the CI
    alpha = 1.0 - ci
    lower = float(np.percentile(bootstrap_means, (alpha / 2) * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
    
    # Check for edge cases
    if np.isnan(lower) or np.isnan(upper):
        # Fallback to normal approximation if bootstrap fails
        from scipy import stats
        se_naive = np.std(scores_arr, ddof=1) / np.sqrt(n)
        z = stats.norm.ppf(1 - alpha / 2)
        lower = actual_mean - z * se_naive
        upper = actual_mean + z * se_naive
        method = "normal_approximation"
    else:
        method = "percentile"
    
    return {
        "lower": round(lower, 4),
        "mean": round(actual_mean, 4),
        "upper": round(upper, 4),
        "width": round(upper - lower, 4),
        "n": n,
        "se": round(se, 6),
        "method": method,
    }


# ── CALIBRATION DATA ─────────────────────────────────────────────────────────

def calibration_data(confidences: list, accuracies: list, n_bins: int = 10) -> dict:
    """
    Compute calibration curve data for the router's confidence scores.

    Use case: When the router says "90% confident this is an order query",
    is it actually correct 90% of the time? Perfect calibration = diagonal.

    How it works:
      1. Group predictions into confidence buckets (0-10%, 10-20%, ..., 90-100%)
      2. For each bucket, compute the average confidence and actual accuracy
      3. Plot avg_confidence vs actual_accuracy — deviation from diagonal = miscalibration

    Args:
        confidences: List of router confidence scores (0.0 to 1.0)
        accuracies:  List of binary correctness flags (1 = correct, 0 = wrong)
        n_bins:      Number of confidence buckets

    Returns:
        Dict with:
          - bins:           List of dicts with bin_start, bin_end, avg_confidence,
                            actual_accuracy, count
          - ece:            Expected Calibration Error (lower = better calibrated)
          - interpretation: Human-readable summary
    """
    if len(confidences) != len(accuracies):
        return {
            "bins": [],
            "ece": 0.0,
            "interpretation": "ERROR: Confidences and accuracies must have equal length.",
        }

    if not confidences:
        return {
            "bins": [],
            "ece": 0.0,
            "interpretation": "No data provided.",
        }

    conf_arr = np.array(confidences, dtype=float)
    acc_arr = np.array(accuracies, dtype=float)
    n_total = len(conf_arr)

    bins = []
    ece = 0.0  # Expected Calibration Error

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes the right edge
            mask = (conf_arr >= bin_start) & (conf_arr <= bin_end)
        else:
            mask = (conf_arr >= bin_start) & (conf_arr < bin_end)

        count = int(np.sum(mask))

        if count > 0:
            avg_conf = float(np.mean(conf_arr[mask]))
            actual_acc = float(np.mean(acc_arr[mask]))
            ece += (count / n_total) * abs(actual_acc - avg_conf)
        else:
            avg_conf = (bin_start + bin_end) / 2
            actual_acc = None

        bins.append({
            "bin_start": round(bin_start, 2),
            "bin_end": round(bin_end, 2),
            "avg_confidence": round(avg_conf, 4),
            "actual_accuracy": round(actual_acc, 4) if actual_acc is not None else None,
            "count": count,
        })

    # Interpret ECE
    if ece < 0.05:
        cal_quality = "Excellent"
    elif ece < 0.10:
        cal_quality = "Good"
    elif ece < 0.20:
        cal_quality = "Fair"
    else:
        cal_quality = "Poor"

    interpretation = f"{cal_quality} calibration (ECE={ece:.4f}). Lower ECE = better calibrated."

    return {
        "bins": bins,
        "ece": round(float(ece), 4),
        "interpretation": interpretation,
    }


# ── STANDALONE TEST ──────────────────────────────────────────────────────────
# Run: python -m evals.metrics

if __name__ == "__main__":
    print("=" * 60)
    print("Statistical Metrics Module -- Verification Test")
    print("=" * 60)

    all_pass = True

    # --- Test 1: Spearman on identical arrays ---
    print("\n--- Test 1: Spearman (identical arrays) ---")
    r = spearman_correlation([4, 3, 5, 2, 1], [4, 3, 5, 2, 1])
    print(f"  rho={r['rho']}, p={r['p_value']}")
    print(f"  {r['interpretation']}")
    if r["rho"] != 1.0:
        print("  [FAIL] Expected rho=1.0")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 2: Spearman on reversed arrays ---
    print("\n--- Test 2: Spearman (perfectly reversed) ---")
    r = spearman_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    print(f"  rho={r['rho']}, p={r['p_value']}")
    print(f"  {r['interpretation']}")
    if r["rho"] != -1.0:
        print("  [FAIL] Expected rho=-1.0")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 3: Cohen's d (large improvement) ---
    print("\n--- Test 3: Cohen's d (large improvement) ---")
    baseline = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0, 2.0, 2.5]
    current = [4.0, 4.5, 5.0, 4.0, 4.5, 5.0, 4.0, 4.5]
    r = cohens_d(current, baseline)
    print(f"  d={r['d']}")
    print(f"  {r['interpretation']}")
    if abs(r["d"]) < 0.8:
        print("  [FAIL] Expected large effect (d > 0.8)")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 4: Cohen's d (negligible difference) ---
    print("\n--- Test 4: Cohen's d (negligible difference) ---")
    r = cohens_d([4.0, 4.1, 3.9, 4.0], [4.0, 3.9, 4.1, 4.0])
    print(f"  d={r['d']}")
    print(f"  {r['interpretation']}")
    if abs(r["d"]) > 0.2:
        print("  [FAIL] Expected negligible effect (d < 0.2)")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 5: Paired t-test (significant) ---
    print("\n--- Test 5: Paired t-test (significant difference) ---")
    r = paired_ttest(current, baseline)
    print(f"  t={r['t_stat']}, p={r['p_value']}")
    print(f"  {r['interpretation']}")
    if r["p_value"] >= 0.05:
        print("  [FAIL] Expected p < 0.05")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 6: Cohen's kappa (perfect agreement) ---
    print("\n--- Test 6: Cohen's kappa (perfect agreement) ---")
    r = cohens_kappa(
        ["PASS", "FAIL", "PASS", "PASS", "FAIL"],
        ["PASS", "FAIL", "PASS", "PASS", "FAIL"],
    )
    print(f"  kappa={r['kappa']}")
    print(f"  {r['interpretation']}")
    if r["kappa"] != 1.0:
        print("  [FAIL] Expected kappa=1.0")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 7: Cohen's kappa (partial agreement) ---
    print("\n--- Test 7: Cohen's kappa (partial agreement) ---")
    r = cohens_kappa(
        ["PASS", "FAIL", "PASS", "PASS", "FAIL", "PASS", "FAIL", "PASS"],
        ["PASS", "FAIL", "FAIL", "PASS", "FAIL", "PASS", "PASS", "PASS"],
    )
    print(f"  kappa={r['kappa']}")
    print(f"  {r['interpretation']}")
    if r["kappa"] >= 1.0 or r["kappa"] <= 0.0:
        print("  [FAIL] Expected partial agreement (0 < kappa < 1)")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 8: Bootstrap CI (tight interval) ---
    print("\n--- Test 8: Bootstrap CI (tight data) ---")
    r = bootstrap_ci([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    print(f"  lower={r['lower']}, mean={r['mean']}, upper={r['upper']}, width={r['width']}")
    if r["width"] > 0.01:
        print("  [FAIL] Expected tight CI for constant data")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 9: Bootstrap CI (spread data) ---
    print("\n--- Test 9: Bootstrap CI (spread data) ---")
    r = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 5.0])
    print(f"  lower={r['lower']}, mean={r['mean']}, upper={r['upper']}, width={r['width']}")
    if r["width"] < 0.5:
        print("  [FAIL] Expected wider CI for spread data")
        all_pass = False
    else:
        print("  [PASS]")

    # --- Test 10: Calibration (perfect calibration) ---
    print("\n--- Test 10: Calibration (well-calibrated) ---")
    # Simulated: high confidence = correct, low confidence = wrong
    confs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.30, 0.25, 0.20, 0.15, 0.10]
    accs =  [1,    1,    1,    1,    1,    0,    0,    0,    0,    0   ]
    r = calibration_data(confs, accs, n_bins=5)
    print(f"  ECE={r['ece']}")
    print(f"  {r['interpretation']}")
    filled_bins = [b for b in r["bins"] if b["count"] > 0]
    print(f"  Non-empty bins: {len(filled_bins)}")
    for b in filled_bins:
        print(f"    [{b['bin_start']:.1f}-{b['bin_end']:.1f}] "
              f"conf={b['avg_confidence']:.2f} acc={b['actual_accuracy']:.2f} "
              f"n={b['count']}")
    print("  [PASS]" if r["ece"] < 0.3 else "  [FAIL]")
    if r["ece"] >= 0.3:
        all_pass = False

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] All statistical metrics verified successfully!")
    else:
        print("[WARN] Some tests failed -- review above.")
