"""Statistical tests for session-level hallucination evaluation.

Chi-square: test independence of adoption rates across protocols/domains.
Kruskal-Wallis: non-parametric comparison of collapse scores across groups.
Dunn's post-hoc: pairwise comparisons after significant Kruskal-Wallis.
Wilson CI: binomial confidence intervals for adoption rates.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import chi2_contingency, kruskal, norm


def wilson_confidence_interval(successes, total, z=1.96):
    """Wilson score interval for binomial proportion.
    
    Returns (lower, upper, center).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
    return max(0, center - margin), min(1, center + margin), p


def chi_square_test(contingency_dict):
    """Chi-square test for independence across groups.
    
    Args:
        contingency_dict: {group_label: [adoption_count, rejection_count]}
    
    Returns:
        (chi2, p_value, dof) or (None, None, None) if insufficient data.
    """
    rows = []
    for key, (adopt, reject) in contingency_dict.items():
        total = adopt + reject
        if total < 5:
            continue
        rows.append([adopt, reject])
    
    if len(rows) < 2:
        return None, None, None
    
    table = np.array(rows)
    try:
        chi2, p, dof, _ = chi2_contingency(table)
        return round(chi2, 3), round(p, 4), dof
    except ValueError:
        return None, None, None


def kruskal_wallis_test(grouped_scores):
    """Kruskal-Wallis H-test comparing collapse scores across groups.

    Args:
        grouped_scores: {group_label: [score1, score2, ...]}
    
    Returns:
        (H_statistic, p_value) or (None, None) if <2 groups.
    """
    groups = [scores for scores in grouped_scores.values() if len(scores) >= 5]
    if len(groups) < 2:
        return None, None
    try:
        h, p = kruskal(*groups)
        return round(h, 3), round(p, 4)
    except (ValueError, TypeError):
        return None, None


def dunn_posthoc(grouped_scores, alpha=0.05):
    """Dunn's post-hoc test with Bonferroni correction.
    
    Args:
        grouped_scores: {group_label: [score1, score2, ...]}
        alpha: significance level
    
    Returns:
        [(group_a, group_b, z_score, p_corrected, significant)]
    """
    groups = {k: v for k, v in grouped_scores.items() if len(v) >= 5}
    labels = sorted(groups.keys())
    if len(labels) < 2:
        return []
    
    # Compute ranks
    all_scores = []
    for label in labels:
        all_scores.extend((s, label) for s in groups[label])
    all_scores.sort(key=lambda x: x[0])
    
    ranks = defaultdict(list)
    for rank_idx, (_, label) in enumerate(all_scores, start=1):
        ranks[label].append(rank_idx)
    
    mean_ranks = {label: np.mean(ranks[label]) for label in labels}
    n_total = len(all_scores)
    
    results = []
    pairs = list(combinations(labels, 2))
    n_tests = len(pairs)
    
    for a, b in pairs:
        n_a, n_b = len(groups[a]), len(groups[b])
        r_a, r_b = mean_ranks[a], mean_ranks[b]
        
        denominator = np.sqrt((n_total * (n_total + 1) / 12.0) * (1.0 / n_a + 1.0 / n_b))
        if denominator == 0:
            z = 0.0
        else:
            z = (r_a - r_b) / denominator
        
        # Bonferroni correction
        p_uncorrected = 2 * (1 - norm.cdf(abs(z)))
        p_corrected = min(p_uncorrected * n_tests, 1.0)
        
        results.append({
            "group_a": a,
            "group_b": b,
            "z_score": round(z, 3),
            "p_corrected": round(p_corrected, 4),
            "significant": p_corrected < alpha,
        })
    
    return results


def compute_all_statistics(judged_results):
    """Compute all statistical tests on judged data.
    
    Args:
        judged_results: list of (turn_record, track1, track2, justification)
    
    Returns:
        dict with keys: chi_square, kruskal, dunn, protocol_ci, domain_ci
    """
    turns_with_scores = [(t, t1, t2) for t, t1, t2, _ in judged_results if t["turn"] >= 5]
    
    # Group by protocol
    proto_track1 = defaultdict(lambda: [0, 0])  # [adopt, reject]
    proto_track2 = defaultdict(list)
    domain_track1 = defaultdict(lambda: [0, 0])
    domain_track2 = defaultdict(list)
    
    for t, t1, t2 in turns_with_scores:
        case_id = t.get("caseId", "?")
        proto = t.get("protocol", "?")
        
        if t1 is not None:
            key = proto.split("_protocol_", 1)[1] if "_protocol_" in proto else proto
            if t1 == 1:
                proto_track1[key][0] += 1
            else:
                proto_track1[key][1] += 1
            proto_track2[key].append(t2)
        
        if t1 is not None and t2 is not None:
            domain_track1[case_id][0 if t1 == 1 else 1] += 0 if t1 == 1 else 0
            # fix: just count
            if t1 == 1:
                domain_track1[case_id][0] += 1
            else:
                domain_track1[case_id][1] += 1
            domain_track2[case_id].append(t2)
    
    # Chi-square on protocol contingency table
    chi2, chi_p, chi_dof = chi_square_test(proto_track1)
    
    # Kruskal-Wallis on protocol track2 scores
    kw_h, kw_p = kruskal_wallis_test(proto_track2)
    
    # Dunn's post-hoc
    dunn = dunn_posthoc(proto_track2)
    
    # Protocol-level CI
    proto_ci = {}
    for proto, (adopt, reject) in proto_track1.items():
        lo, hi, rate = wilson_confidence_interval(adopt, adopt + reject)
        proto_ci[proto] = {
            "adoptions": adopt,
            "total": adopt + reject,
            "rate": round(rate, 3),
            "ci_lower": round(lo, 3),
            "ci_upper": round(hi, 3),
        }
    
    # Domain-level CI
    domain_ci = {}
    for domain, (adopt, reject) in domain_track1.items():
        lo, hi, rate = wilson_confidence_interval(adopt, adopt + reject)
        domain_ci[domain] = {
            "adoptions": adopt,
            "total": adopt + reject,
            "rate": round(rate, 3),
            "ci_lower": round(lo, 3),
            "ci_upper": round(hi, 3),
        }
    
    return {
        "chi_square": {"statistic": chi2, "p_value": chi_p, "dof": chi_dof},
        "kruskal_wallis": {"h_statistic": kw_h, "p_value": kw_p},
        "dunn_posthoc": dunn,
        "protocol_ci": proto_ci,
        "domain_ci": domain_ci,
    }
