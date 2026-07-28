"""
training/train.py
==================
Top-level training orchestration for AMP and CPP classifiers + CVAE generator.

CORE DESIGN — classifier vs. generator use different positive sets:
  The classifier trains on a 3:1 pos:neg matched subsample
  (amp_pos_clf.csv / cpp_pos_clf.csv). At 27:1 (the full positive pool)
  the AMP classifier scored random peptides 0.862 — non-discriminative.
  3:1 is the regime where the CPP classifier is known to work (0.392).
  The generator (CVAE) has no such constraint and trains on the full
  HemoPI2-filtered pool (amp_generator_positives.csv /
  cpp_generator_positives.csv) for maximum sequence diversity.
  See train_generator() below and data.loader.load_generator_positives().

  AMP negatives are DBAASP peptides (MIC >=256 ug/mL, >=3 species,
  100% experimentally measured) — not UniProt. UniProt cytosolic proteins
  were evaluated and rejected as an AMP negative source.

Evaluation reporting:
- CV ROC-AUC (training fold only)
- Validation ROC-AUC, PR-AUC, Brier score (calibration quality)
- Test ROC-AUC, PR-AUC, Brier score (final, call once)
- Calibration curves saved to output directory
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd

from data.loader import (
    build_amp_dataset, build_cpp_dataset, stratified_split
)
from models.classifier import train_predictor, evaluate_on_test

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: save evaluation metrics
# =============================================================================

def _save_metrics(metrics: dict, path: str):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


# =============================================================================
# AMP classifier
# =============================================================================

def train_amp_classifier(
    amp_pos_path: str,
    amp_neg_path: Optional[str],
    output_dir: str,
    data_cfg: dict,
    model_cfg: dict,
    seed: int = 42,
) -> Tuple[Callable, Callable, pd.DataFrame]:
    """
    Train, calibrate, and evaluate the AMP classifier.

    Returns (predict_calibrated, predict_raw, amp_train_df).
      predict_calibrated — isotonic-calibrated probability. Use for
        reporting and any final candidate score.
      predict_raw — uncalibrated ensemble average. Use for generation
        steering — calibration on a ~200-sequence validation split can
        collapse to a coarse step function with no usable gradient for
        softmax-weighted seed selection. See models/classifier.py
        train_predictor() docstring for the full explanation.
    amp_train_df is the classifier's training-split positives — kept
    for diagnostics only. It is NOT used to train the CVAE generator;
    the generator trains on the separate, much larger
    amp_generator_positives.csv pool (see data.load_generator_positives()
    and train_generator()).
    """
    logger.info("=" * 60)
    logger.info("AMP CLASSIFIER TRAINING")
    logger.info("=" * 60)

    dataset = build_amp_dataset(
        pos_path=amp_pos_path,
        neg_path=amp_neg_path,
        min_len=data_cfg.get('amp_min_len', 5),
        max_len=data_cfg.get('amp_max_len', 50),
    )

    train_df, val_df, test_df = stratified_split(
        dataset,
        val_frac=data_cfg.get('val_frac', 0.15),
        test_frac=data_cfg.get('test_frac', 0.15),
        seed=seed,
    )

    predict_amp_cal, predict_amp_raw, val_metrics = train_predictor(
        train_seqs=train_df['sequence'].tolist(),
        train_labels=train_df['label'].values,
        val_seqs=val_df['sequence'].tolist(),
        val_labels=val_df['label'].values,
        cv_folds=model_cfg.get('cv_folds', 5),
        seed=seed,
        calibration_method=model_cfg.get('calibration_method', 'isotonic'),
    )

    # Test-set metrics (ROC-AUC, PR-AUC, Brier, calibration curve) must use
    # the CALIBRATED function — these are meaningless on raw scores.
    test_metrics = evaluate_on_test(
        predict_amp_cal,
        test_df['sequence'].tolist(),
        test_df['label'].values,
    )

    all_metrics = {**val_metrics, **test_metrics}
    _save_metrics(all_metrics, os.path.join(output_dir, 'amp_metrics.json'))

    logger.info("AMP TEST ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
                test_metrics['test_roc_auc'],
                test_metrics['test_pr_auc'],
                test_metrics['test_brier'])

    # Return training positives for diagnostics (not used by the CVAE)
    amp_train_pos = train_df[train_df['label'] == 1].reset_index(drop=True)
    return predict_amp_cal, predict_amp_raw, amp_train_pos


# =============================================================================
# CPP classifier
# =============================================================================

def train_cpp_classifier(
    cpp_path: str,
    cpp_neg_path: Optional[str],
    output_dir: str,
    data_cfg: dict,
    model_cfg: dict,
    seed: int = 42,
) -> Tuple[Callable, Callable]:
    """
    Train, calibrate, and evaluate the CPP classifier.

    Returns (predict_calibrated, predict_raw) — see train_amp_classifier()
    docstring for why both are needed.
    """
    logger.info("=" * 60)
    logger.info("CPP CLASSIFIER TRAINING")
    logger.info("=" * 60)

    dataset = build_cpp_dataset(
        cpp_path=cpp_path,
        neg_path=cpp_neg_path,
        min_len=data_cfg.get('cpp_min_len', 5),
        max_len=data_cfg.get('cpp_max_len', 30),
    )

    train_df, val_df, test_df = stratified_split(
        dataset,
        val_frac=data_cfg.get('val_frac', 0.15),
        test_frac=data_cfg.get('test_frac', 0.15),
        seed=seed,
    )

    predict_cpp_cal, predict_cpp_raw, val_metrics = train_predictor(
        train_seqs=train_df['sequence'].tolist(),
        train_labels=train_df['label'].values,
        val_seqs=val_df['sequence'].tolist(),
        val_labels=val_df['label'].values,
        cv_folds=model_cfg.get('cv_folds', 5),
        seed=seed,
        calibration_method=model_cfg.get('calibration_method', 'isotonic'),
    )

    test_metrics = evaluate_on_test(
        predict_cpp_cal,
        test_df['sequence'].tolist(),
        test_df['label'].values,
    )

    all_metrics = {**val_metrics, **test_metrics}
    _save_metrics(all_metrics, os.path.join(output_dir, 'cpp_metrics.json'))

    logger.info("CPP TEST ROC-AUC: %.3f  PR-AUC: %.3f  Brier: %.4f",
                test_metrics['test_roc_auc'],
                test_metrics['test_pr_auc'],
                test_metrics['test_brier'])

    return predict_cpp_cal, predict_cpp_raw


# =============================================================================
# CVAE generator
# =============================================================================

def train_generator(
    seqs: list,
    generator_cfg: dict,
    seed: int = 42,
):
    """
    Train the CVAE on the generator-positives pool.

    CORE DESIGN: `seqs` must come from amp_generator_positives.csv /
    cpp_generator_positives.csv (via data.loader.load_generator_positives()),
    NOT from the classifier's train split. The classifier trains on a
    3:1-matched subsample (1,005 AMP / 651 CPP positives); the generator
    needs the full HemoPI2-filtered pool (9,205 AMP / 652 CPP) for sequence
    diversity. Passing classifier positives here silently starves the CVAE
    down to ~700 training sequences instead of ~9,800 — this was a real
    regression in an earlier version of this pipeline. Caller (main.py) is
    responsible for loading and combining the AMP+CPP generator pools before
    calling this function; this function trains a single CVAE on whatever
    list it is given.

    Selects PyTorch backend if available; falls back to NumPy.
    """
    logger.info("=" * 60)
    logger.info("CVAE GENERATOR TRAINING")
    logger.info("=" * 60)
    logger.info("Generator training pool size: %d sequences", len(seqs))

    lengths = [len(s) for s in seqs]
    max_len = int(np.percentile(lengths, 90))
    max_len = min(max(max_len, 15), generator_cfg.get('max_len', 50))

    logger.info("Training CVAE on %d sequences | max_len=%d", len(seqs), max_len)

    backend = generator_cfg.get('backend', 'torch')
    use_torch = (backend == 'torch')

    if use_torch:
        try:
            import torch  # noqa: F401 — just verify it's available
            from models.cvae_torch import train_cvae
            logger.info("Using PyTorch CVAE backend")
            model = train_cvae(
                seqs=seqs,
                max_len=max_len,
                latent_dim=generator_cfg.get('latent_dim', 64),
                hidden_dim=generator_cfg.get('hidden_dim', 256),
                epochs=generator_cfg.get('epochs', 150),
                batch_size=generator_cfg.get('batch_size', 64),
                lr=generator_cfg.get('lr', 1e-3),
                grad_clip=generator_cfg.get('grad_clip', 5.0),
                kl_beta=generator_cfg.get('kl_beta', 1.0),
                kl_warmup_epochs=generator_cfg.get('kl_warmup_epochs', 60),
                checkpoint_dir=generator_cfg.get('checkpoint_dir', None),
                checkpoint_every=generator_cfg.get('checkpoint_every', 25),
                resume_from=generator_cfg.get('resume_from', None),
                device='cpu',
            )
            return model
        except ImportError:
            logger.warning("PyTorch not available; falling back to NumPy CVAE")

    logger.info("Using NumPy CVAE backend")
    from models.cvae_numpy import PeptideCVAENP
    np.random.seed(seed)
    model = PeptideCVAENP(
        max_len=max_len,
        latent_dim=generator_cfg.get('latent_dim', 64),
        hidden_dim=generator_cfg.get('hidden_dim', 256),
        lr=generator_cfg.get('lr', 1e-3),
        seed=seed,
    )
    model.train(
        seqs=seqs,
        epochs=generator_cfg.get('epochs', 150),
        batch_size=generator_cfg.get('batch_size', 64),
        kl_beta=generator_cfg.get('kl_beta', 1.0),
        kl_warmup_epochs=generator_cfg.get('kl_warmup_epochs', 60),
        grad_clip=generator_cfg.get('grad_clip', 5.0),
        checkpoint_dir=generator_cfg.get('checkpoint_dir', None),
        checkpoint_every=generator_cfg.get('checkpoint_every', 25),
        resume_from=generator_cfg.get('resume_from', None),
    )
    return model
