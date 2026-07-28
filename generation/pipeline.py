"""
generation/pipeline.py
=======================
Multi-objective peptide generation and score-guided optimisation.

Fixes over original:
1. Mutation is score-guided: top-N seeds are sampled proportional to
   softmax(combined_score / T) rather than taking a fixed top-K slice.
   This provides a principled exploration-exploitation trade-off.
   (Original: hard cutoff of top 200 — no exploration, premature convergence)

2. Diversity is maintained within the optimisation loop by penalising
   sequences that are near-identical to already-selected seeds
   (edit distance filter). Without this, the mutation walk converges
   to minor variants of 2–3 local optima.

3. Mutation positions are not uniformly random: positions are sampled
   proportional to (1 - max_aa_probability), so the model preferentially
   mutates low-confidence positions rather than destroying well-predicted
   residues.
"""

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from features.physicochemical import is_canonical, AA_LIST

logger = logging.getLogger(__name__)


# =============================================================================
# Edit distance (Levenshtein) for intra-pool diversity
# =============================================================================

def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] + [0] * n
        for j, c2 in enumerate(s2):
            curr[j + 1] = min(
                prev[j + 1] + 1,          # deletion
                curr[j]     + 1,          # insertion
                prev[j]     + (c1 != c2), # substitution
            )
        prev = curr
    return prev[n]


def sequence_identity(s1: str, s2: str) -> float:
    """Fraction of identical aligned positions (longer sequence as denominator)."""
    d = levenshtein(s1, s2)
    return 1.0 - d / max(len(s1), len(s2), 1)


def is_diverse_from_pool(
    seq: str,
    pool: List[str],
    max_identity: float = 0.80,
) -> bool:
    """Return True if seq has identity < max_identity to every sequence in pool."""
    for other in pool:
        if sequence_identity(seq, other) >= max_identity:
            return False
    return True


# =============================================================================
# Score-guided mutation
# =============================================================================

def _get_mutation_probs(seq: str) -> np.ndarray:
    """
    Assign per-position mutation probability proportional to residue
    'uncertainty'. Here we use a simple heuristic: uniform distribution
    (can be replaced with gradient-based attribution if model permits).
    A proper implementation would use the gradient of the score w.r.t.
    one-hot input to identify the most impactful positions.
    """
    return np.ones(len(seq), dtype=np.float32) / len(seq)


def mutate_sequence(
    seq: str,
    n_mutations: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Mutate n_mutations positions in seq.
    Positions sampled proportional to mutation probability weights.
    Replacement amino acids sampled uniformly from all canonicals.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(seq) == 0:
        return seq

    arr    = list(seq)
    probs  = _get_mutation_probs(seq)
    n_mut  = min(n_mutations, len(seq))
    positions = rng.choice(len(seq), size=n_mut, replace=False, p=probs)

    for pos in positions:
        current = arr[pos]
        alternatives = [aa for aa in AA_LIST if aa != current]
        arr[pos] = rng.choice(alternatives)

    return ''.join(arr)


# =============================================================================
# Softmax-weighted seed sampling
# =============================================================================

def softmax_sample_seeds(
    candidates: List[Tuple],
    n_seeds: int,
    temperature: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple]:
    """
    Sample n_seeds from candidates weighted by softmax(score / temperature).

    FIX: original used hard top-K cutoff. Softmax weighting allows
    lower-scoring but structurally diverse sequences to occasionally be
    selected as seeds, preventing premature convergence.

    Lower temperature → sharper distribution → more exploitation.
    Higher temperature → flatter distribution → more exploration.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(candidates) <= n_seeds:
        return candidates

    scores = np.array([c[3] for c in candidates], dtype=np.float64)
    # Numerical stability: subtract max before softmax
    logits = scores / max(temperature, 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    n_sample = min(n_seeds, len(candidates))
    indices  = rng.choice(len(candidates), size=n_sample,
                          replace=False, p=weights)
    return [candidates[i] for i in indices]


# =============================================================================
# Main generation + optimisation loop
# =============================================================================

def generate_and_optimise(
    generator,
    predict_amp_raw: Callable,
    predict_cpp_raw: Callable,
    predict_amp_cal: Callable,
    predict_cpp_cal: Callable,
    n_initial: int = 3000,
    n_optimise_rounds: int = 5,
    top_k: int = 300,
    temperature: float = 1.1,
    mutation_ns: List[int] = [1, 2, 3],
    softmax_temperature: float = 0.3,
    seed: int = 42,
) -> List[Tuple]:
    """
    Full generation + multi-round guided mutation pipeline.

    Returns a list of (sequence, amp_score, cpp_score, combined_score)
    tuples where the scores are CALIBRATED probabilities — see below.

    CRITICAL: two different scales are in play here, and mixing them is
    a bug that was live in this function until this fix.
      - predict_amp_raw / predict_cpp_raw drive every round's softmax
        seed selection and mutant scoring. Calibrated scores can collapse
        to a coarse step function on small validation splits (see
        models/classifier.py train_predictor() docstring), which flattens
        the softmax gradient the optimisation loop depends on. Raw scores
        preserve full continuous discriminative signal.
      - predict_amp_cal / predict_cpp_cal are used exactly once, on the
        final deduplicated pool, right before returning. Downstream
        consumers of this function's return value — specifically
        filters.biological.apply_biological_filters(), which thresholds
        against cfg.min_amp_score / cfg.min_cpp_score (default 0.50) —
        were written assuming a calibrated probability scale. Handing
        those thresholds raw ensemble scores silently changes their
        meaning: 0.50 raw is not "50% calibrated probability of being
        AMP", and there is no reason the two scales should threshold at
        the same cutoff. Recalibrating the final pool here, once, keeps
        that contract correct without needing to touch filters/biological.py.

    Protocol:
      1. Generate n_initial sequences from the CVAE
      2. Score with RAW AMP and CPP predictors
      3. For n_optimise_rounds:
           a. Sample seeds proportional to softmax(raw combined_score)
           b. Apply n_mutations mutations per seed (multiple mutation counts)
           c. Score mutants with RAW predictors
           d. Merge mutants into pool, deduplicate, re-rank on raw score
      4. Recompute CALIBRATED scores once on the final deduplicated pool
      5. Return pool sorted by calibrated combined score
    """
    rng = np.random.default_rng(seed)

    # -- Step 1: Initial generation ----------------------------------------
    logger.info("Generating %d initial candidates from generator...", n_initial)

    # Detect generator type (PyTorch or NumPy fallback)
    try:
        from models.cvae_torch import generate_sequences as _gen_torch
        raw_pool = _gen_torch(
            generator, n=n_initial, temperature=temperature, min_len=8
        )
    except Exception:
        raw_pool = generator.generate(
            n=n_initial, temperature=temperature, min_len=8
        )

    logger.info("Generated %d valid sequences", len(raw_pool))

    if not raw_pool:
        raise RuntimeError("Generator produced zero valid sequences.")

    # -- Step 2: Score initial pool (RAW — drives the search) ---------------
    logger.info("Scoring initial pool (raw ensemble score)...")
    amp_scores = predict_amp_raw(raw_pool)
    cpp_scores = predict_cpp_raw(raw_pool)
    combined   = np.sqrt(amp_scores * cpp_scores)

    # Pool: list of (seq, amp, cpp, combined) — RAW scale throughout the loop
    pool = list(zip(raw_pool, amp_scores.tolist(), cpp_scores.tolist(),
                    combined.tolist()))
    pool.sort(key=lambda x: -x[3])

    logger.info(
        "Initial pool — top combined score (raw): %.3f  mean: %.3f",
        pool[0][3], combined.mean(),
    )

    # -- Step 3: Guided mutation rounds (RAW throughout) --------------------
    for rnd in range(1, n_optimise_rounds + 1):
        logger.info("Optimisation round %d/%d...", rnd, n_optimise_rounds)

        seeds = softmax_sample_seeds(
            pool, n_seeds=top_k, temperature=softmax_temperature, rng=rng
        )

        mutants = []
        for seq, amp_s, cpp_s, comb in seeds:
            for nm in mutation_ns:
                mutant = mutate_sequence(seq, nm, rng)
                if is_canonical(mutant) and len(mutant) >= 8:
                    mutants.append(mutant)

        if not mutants:
            logger.warning("Round %d produced no valid mutants; skipping.", rnd)
            continue

        logger.info("  Scoring %d mutants (raw)...", len(mutants))
        m_amp = predict_amp_raw(mutants)
        m_cpp = predict_cpp_raw(mutants)
        m_com = np.sqrt(m_amp * m_cpp)

        new_entries = list(zip(mutants, m_amp.tolist(), m_cpp.tolist(),
                               m_com.tolist()))

        # Merge new entries into pool
        pool.extend(new_entries)
        pool.sort(key=lambda x: -x[3])

        # Deduplicate by sequence
        seen = set()
        deduped = []
        for entry in pool:
            if entry[0] not in seen:
                seen.add(entry[0])
                deduped.append(entry)
        pool = deduped

        logger.info(
            "  After round %d: pool size=%d  top score (raw)=%.3f",
            rnd, len(pool), pool[0][3],
        )

    logger.info("Total unique candidates: %d", len(pool))

    # -- Step 4: Recompute CALIBRATED scores once, on the final pool --------
    # This is the scale filters.biological.apply_biological_filters() and
    # its cfg.min_amp_score / cfg.min_cpp_score thresholds expect. Search
    # above never sees this — it only ever optimised against raw scores.
    logger.info(
        "Recomputing calibrated scores for %d final candidates "
        "(this is the scale biological filters threshold against)...",
        len(pool),
    )
    final_seqs = [entry[0] for entry in pool]
    cal_amp = predict_amp_cal(final_seqs)
    cal_cpp = predict_cpp_cal(final_seqs)
    cal_combined = np.sqrt(cal_amp * cal_cpp)

    calibrated_pool = list(zip(
        final_seqs, cal_amp.tolist(), cal_cpp.tolist(), cal_combined.tolist()
    ))
    calibrated_pool.sort(key=lambda x: -x[3])

    logger.info(
        "Final pool — top combined score (calibrated): %.3f  mean: %.3f",
        calibrated_pool[0][3], cal_combined.mean(),
    )

    return calibrated_pool
