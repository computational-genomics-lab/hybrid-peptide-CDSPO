"""
training/evaluate.py
=====================
Evaluation of generated peptides against reference distributions.

Provides:
- Score distribution comparison (generated vs. training positives vs.
  training negatives vs. random)
- Novelty metrics (edit distance to nearest training sequence)
- Baseline: random peptide population at the same length distribution
"""

import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from generation.pipeline import levenshtein
from features.physicochemical import AA_LIST, is_canonical

logger = logging.getLogger(__name__)


# =============================================================================
# Random baseline generation
# =============================================================================

def generate_random_peptides(
    n: int,
    length_distribution: List[int],
    seed: int = 42,
) -> List[str]:
    """
    Generate n random peptides drawn from uniform AA distribution
    at lengths sampled from length_distribution.
    Used as a null-hypothesis baseline.
    """
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        L = int(rng.choice(length_distribution))
        seqs.append(''.join(rng.choice(AA_LIST, size=L)))
    return seqs


# =============================================================================
# Novelty metrics
# =============================================================================

def nearest_training_distance(
    generated_seqs: List[str],
    training_seqs: List[str],
    n_sample: Optional[int] = None,
) -> np.ndarray:
    """
    For each generated sequence, compute the normalised edit distance
    to the nearest training sequence.

    Returns an array of shape (len(generated_seqs),) with values in [0, 1].
    0.0 = identical to a training sequence (memorised)
    1.0 = maximally different

    n_sample: if provided, subsamples the training set to this many
    reference sequences for speed. Default is None — the FULL training
    set is used. A memorisation/novelty claim computed against a random
    200-sequence subsample of a much larger training pool is not
    defensible: a generated sequence could be a near-exact copy of a
    training sequence that simply wasn't in the sample, understating
    memorisation. Pass n_sample explicitly to opt into subsampling on a
    very large training pool where full comparison is too slow.
    """
    rng = np.random.default_rng(0)
    if n_sample is not None and len(training_seqs) > n_sample:
        ref = list(rng.choice(len(training_seqs), size=n_sample, replace=False))
        ref_seqs = [training_seqs[i] for i in ref]
        logger.info(
            "nearest_training_distance: subsampled %d of %d training "
            "sequences (n_sample explicitly set)", n_sample, len(training_seqs),
        )
    else:
        ref_seqs = training_seqs
        logger.info(
            "nearest_training_distance: using full training set (%d sequences) "
            "as reference — this may take a while for large pools.",
            len(ref_seqs),
        )

    distances = []
    for gen_seq in generated_seqs:
        min_dist = min(
            levenshtein(gen_seq, ref) / max(len(gen_seq), len(ref), 1)
            for ref in ref_seqs
        )
        distances.append(min_dist)
    return np.array(distances)


# =============================================================================
# Score distribution comparison
# =============================================================================

def compare_score_distributions(
    generated_seqs: List[str],
    training_pos_seqs: List[str],
    predict_amp: Callable,
    predict_cpp: Callable,
    amp_training_neg_seqs: Optional[List[str]] = None,
    cpp_training_neg_seqs: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    n_random: int = 500,
) -> pd.DataFrame:
    """
    Compare AMP and CPP score distributions across up to five populations:
    1. Generated sequences (from CVAE + optimisation)
    2. Training positives (experimental AMPs used for generator training)
    3. AMP training negatives (DBAASP MIC>=256 ug/mL against >=3 species —
       same file as amp_neg_path). Optional: pass None to skip.
    4. CPP training negatives (same file as cpp_neg_path). Optional: pass
       None to skip.
    5. Random baseline (uniform AA composition, matched length distribution)

    Reports summary statistics and optionally saves to CSV.
    """
    train_lengths = [len(s) for s in training_pos_seqs]
    random_seqs   = generate_random_peptides(n_random, train_lengths, seed=0)
    random_seqs   = [s for s in random_seqs if is_canonical(s)]

    populations = [
        ('generated', generated_seqs),
        ('training_positives', training_pos_seqs[:n_random]),
    ]
    if amp_training_neg_seqs:
        populations.append(('amp_training_negatives', amp_training_neg_seqs[:n_random]))
    if cpp_training_neg_seqs:
        populations.append(('cpp_training_negatives', cpp_training_neg_seqs[:n_random]))
    populations.append(('random_baseline', random_seqs))

    rows = []
    for population, seqs in populations:
        if not seqs:
            continue
        amp_s = predict_amp(seqs)
        cpp_s = predict_cpp(seqs)
        comb  = np.sqrt(amp_s * cpp_s)
        rows.append({
            'population': population,
            'n': len(seqs),
            'amp_mean': float(amp_s.mean()),
            'amp_median': float(np.median(amp_s)),
            'amp_std':  float(amp_s.std()),
            'cpp_mean': float(cpp_s.mean()),
            'cpp_median': float(np.median(cpp_s)),
            'cpp_std':  float(cpp_s.std()),
            'combined_mean': float(comb.mean()),
            'combined_median': float(np.median(comb)),
        })
        logger.info(
            "%-22s  n=%4d  AMP=%.3f±%.3f  CPP=%.3f±%.3f  Comb=%.3f",
            population, len(seqs),
            amp_s.mean(), amp_s.std(),
            cpp_s.mean(), cpp_s.std(),
            comb.mean(),
        )

    df = pd.DataFrame(rows)
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info("Score comparison saved: %s", output_path)
    return df


# =============================================================================
# Novelty summary
# =============================================================================

def novelty_report(
    generated_seqs: List[str],
    training_seqs: List[str],
    output_path: Optional[str] = None,
) -> dict:
    """
    Compute novelty statistics for generated sequences.

    generated_seqs MUST be the FINAL candidate set (post-filter,
    post-diversity-control — what actually ends up in candidates.csv),
    not an intermediate generation/mutation pool. Reporting novelty on
    an unfiltered intermediate pool describes sequences the user never
    sees and overstates or understates the novelty of what's actually
    delivered, depending on how the filters correlate with memorisation.

    A sequence with distance 0.0 to the training set was memorised.
    High novelty (distance > 0.5) with high scores indicates genuine
    generation beyond interpolation.
    """
    distances = nearest_training_distance(generated_seqs, training_seqs)
    memorised = (distances < 0.05).sum()
    novel     = (distances > 0.5).sum()

    report = {
        'n_generated': len(generated_seqs),
        'n_memorised': int(memorised),
        'n_novel':     int(novel),
        'pct_memorised': round(100.0 * memorised / max(len(generated_seqs), 1), 2),
        'pct_novel':     round(100.0 * novel     / max(len(generated_seqs), 1), 2),
        'mean_distance': float(distances.mean()),
        'min_distance':  float(distances.min()),
        'p25_distance':  float(np.percentile(distances, 25)),
        'median_distance': float(np.median(distances)),
    }

    logger.info(
        "Novelty: mean dist=%.3f | memorised=%d (%.1f%%) | novel=%d (%.1f%%)",
        report['mean_distance'],
        report['n_memorised'], report['pct_memorised'],
        report['n_novel'],     report['pct_novel'],
    )

    if memorised / max(len(generated_seqs), 1) > 0.20:
        logger.warning(
            "%.1f%% of generated sequences are near-identical to training data. "
            "Consider increasing CVAE temperature or reducing training epochs.",
            report['pct_memorised'],
        )

    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info("Novelty report saved: %s", output_path)

    return report
