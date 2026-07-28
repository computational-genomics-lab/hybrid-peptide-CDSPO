"""
data/loader.py
===============
Data loading, cleaning, and stratified splitting.

CRITICAL FIX (synthetic negatives):
  Original code generated negative class samples by shuffling positive
  sequences. This is biologically invalid: shuffled sequences maintain
  amino acid composition but destroy structural context, and the
  classifier learns to separate by composition rather than by true
  biological signal. Any classifier trained this way will be unable to
  distinguish real non-AMPs/non-CPPs from real AMPs/CPPs with different
  compositions.

  Fix:
  - For CPP: positives (`cpp_pos_clf.csv`) are a curated set of
    experimentally validated CPPs, one 'sequence' column, all label=1.
    Negatives (`cpp_neg_clf.csv`) are 42 benchmark non-CPPs + 175 UniProt
    chain-derived sequences (19% experimental overall), built by
    process_negatives.py. CPPsite 2.0's non-CPP set was evaluated and
    rejected — it is randomly generated from Swiss-Prot, not
    experimentally validated, and must not be used.
  - For AMP: negatives (`amp_neg_clf.csv`) are DBAASP peptides with
    MIC >=256 ug/mL against >=3 species — 100% experimentally measured.
    NOT UniProt-derived. UniProt cytosolic proteins were evaluated and
    rejected as an AMP negative source (see CORE DESIGN notes in
    training/train.py) and must not be substituted or supplemented in.
  - Classifier positives and generator positives are DIFFERENT files
    (`*_clf.csv` vs `*_generator_positives.csv`). This module reads
    both; do not use one in place of the other. See load_generator_positives().

CRITICAL FIX (data split):
  Original code never created a held-out test set. Cross-validation
  was the only evaluation, and calibration was performed on the full
  training data. Fixed by:
  - Stratified 3-way split: train / validation / test
  - Calibration performed on validation set only
  - Test set used exclusively for final metric reporting
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features.physicochemical import is_canonical

logger = logging.getLogger(__name__)


# =============================================================================
# Column name constants (adapt these if your file has different headers)
# =============================================================================
AMP_SEQ_COL = 'Sequence'
AMP_ID_COL = 'SeqID'
CPP_SEQ_COL = 'Protein_Sequence'
CPP_ID_COL = 'Protein_ID'


# =============================================================================
# Low-level file readers
# =============================================================================

def _read_sequence_only_csv(path: str, entity_label: str = 'dataset') -> pd.DataFrame:
    """
    Generic reader for the re-curated one-column CSVs
    (amp_pos_clf.csv, amp_neg_clf.csv, cpp_pos_clf.csv, cpp_neg_clf.csv,
    amp_generator_positives.csv, cpp_generator_positives.csv).

    All six files share the same schema: a single column named 'sequence'.
    Accepts 'Sequence' (capitalised) as a fallback for compatibility with
    older manually-curated files that used that header.

    Returns a DataFrame with a single 'Sequence' column (internal
    convention used throughout this module), regardless of the header
    casing in the source file.
    """
    df = pd.read_csv(path, encoding='latin1')
    df.columns = [c.strip() for c in df.columns]
    if AMP_SEQ_COL in df.columns:
        return df[[AMP_SEQ_COL]]
    for alt in ['sequence', 'seq', 'SEQUENCE', 'Seq']:
        if alt in df.columns:
            return df[[alt]].rename(columns={alt: AMP_SEQ_COL})
    raise ValueError(
        f"{entity_label} file must contain a 'sequence' column. "
        f"Found: {df.columns.tolist()}  (path: {path})"
    )


def _read_amp_positives(path: str) -> pd.DataFrame:
    """
    Load the AMP positive dataset (amp_pos_clf.csv — one 'sequence' column).
    Legacy .xlsx files with 'SeqID'/'Sequence' columns are still supported.
    """
    p = Path(path)
    if p.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        if AMP_SEQ_COL not in df.columns:
            raise ValueError(
                f"AMP file must contain a '{AMP_SEQ_COL}' column. "
                f"Found: {df.columns.tolist()}"
            )
        return df
    return _read_sequence_only_csv(path, entity_label='AMP positives')


def _read_amp_negatives(path: str) -> pd.DataFrame:
    """
    Load non-AMP negative sequences (amp_neg_clf.csv — one 'sequence' column).
    Also reused as the generic reader for cpp_neg_clf.csv, since both files
    share the same single-column schema.

    Source (AMP negatives): DBAASP peptides, MIC >=256 ug/mL against
    >=3 species, 100% experimentally measured. NOT UniProt-derived.
    Source (CPP negatives, when reused for that purpose): 42 benchmark
    non-CPPs + 175 UniProt chain-derived sequences (19% experimental),
    built by process_negatives.py.
    """
    p = Path(path)
    if p.suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        if AMP_SEQ_COL not in df.columns:
            raise ValueError(
                f"Negatives file must contain a '{AMP_SEQ_COL}' column. "
                f"Found: {df.columns.tolist()}"
            )
        return df
    return _read_sequence_only_csv(path, entity_label='Negatives')


def _read_cpp_dataset(path: str) -> pd.DataFrame:
    """
    Load the CPP positive dataset (cpp_pos_clf.csv — one 'sequence' column).

    All rows are experimentally validated CPP positives. This function
    previously parsed the legacy CellPPD multi-column predictor-output
    format; the re-curated cpp_pos_clf.csv is a plain single-column file
    and no longer carries that predictor-output column, so it is read
    with the same generic reader used for AMP positives.
    """
    return _read_sequence_only_csv(path, entity_label='CPP positives')


def load_generator_positives(
    path: str,
    min_len: int = 5,
    max_len: int = 50,
) -> list:
    """
    Load and clean a generator-training positive set
    (amp_generator_positives.csv or cpp_generator_positives.csv).

    CRITICAL: this is a DIFFERENT file from the classifier positives
    (amp_pos_clf.csv / cpp_pos_clf.csv). The generator trains on the full
    HemoPI2-filtered positive pool (9,205 for AMP / 652 for CPP), not the
    3:1-matched classifier subset (1,005 / 651). Do not pass a classifier
    dataset here, and do not pass this output to build_amp_dataset /
    build_cpp_dataset.

    Returns a plain list[str] of cleaned, canonical, length-filtered
    sequences — ready to hand directly to train_generator().
    """
    logger.info("Loading generator positives from %s", path)
    df = _read_sequence_only_csv(path, entity_label='Generator positives')
    df = _clean_sequences(df, AMP_SEQ_COL, min_len, max_len)
    seqs = df[AMP_SEQ_COL].tolist()
    logger.info("Generator positives after cleaning: %d", len(seqs))
    return seqs


def load_negative_sequences(
    path: str,
    min_len: int = 5,
    max_len: int = 50,
    entity_label: str = 'negatives',
) -> list:
    """
    Load and clean a negative-sequence file (amp_neg_clf.csv or
    cpp_neg_clf.csv) as a standalone list, independent of the
    classifier's train/val/test split.

    Use case: evaluation/diagnostics — e.g. checking the classifier's
    score distribution on its own true negatives (see
    training.evaluate.compare_score_distributions()'s training_neg_seqs
    parameter). build_amp_dataset()/build_cpp_dataset() already load
    and split these same sequences for training; this function reads
    the same file again but returns a flat, unsplit list, since the
    training split's val/test negatives aren't individually addressable
    from outside train_amp_classifier()/train_cpp_classifier().

    Returns a plain list[str] of cleaned, canonical, length-filtered
    sequences.
    """
    logger.info("Loading %s from %s", entity_label, path)
    df = _read_amp_negatives(path)
    df = _clean_sequences(df, AMP_SEQ_COL, min_len, max_len)
    seqs = df[AMP_SEQ_COL].tolist()
    logger.info("%s after cleaning: %d", entity_label, len(seqs))
    return seqs


# =============================================================================
# Cleaning
# =============================================================================

def _clean_sequences(
    df: pd.DataFrame,
    seq_col: str,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    df = df.copy()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    df = df.dropna(subset=[seq_col])
    df = df[df[seq_col].apply(is_canonical)]
    df = df[df[seq_col].str.len().between(min_len, max_len)]
    df = df.drop_duplicates(subset=seq_col)
    return df.reset_index(drop=True)


# =============================================================================
# Dataset assemblers
# =============================================================================

def build_amp_dataset(
    pos_path: str,
    neg_path: Optional[str],
    min_len: int = 5,
    max_len: int = 50,
) -> pd.DataFrame:
    """
    Build a labelled AMP dataset.

    Returns a DataFrame with columns ['sequence', 'label'] where
    label=1 → AMP positive, label=0 → non-AMP negative.

    If neg_path is None or the file doesn't exist, the function raises
    a clear error instead of generating synthetic negatives silently.
    """
    logger.info("Loading AMP positives from %s", pos_path)
    pos_df = _read_amp_positives(pos_path)
    pos_df = _clean_sequences(pos_df, AMP_SEQ_COL, min_len, max_len)
    pos_df = pos_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    pos_df['label'] = 1
    logger.info("AMP positives after cleaning: %d", len(pos_df))

    if neg_path is None or not Path(neg_path).exists():
        raise FileNotFoundError(
            "No AMP negative dataset found at path: "
            f"'{neg_path}'.\n"
            "AMP negatives must be DBAASP peptides with MIC >=256 ug/mL\n"
            "against >=3 species (100% experimentally measured). Do NOT\n"
            "substitute UniProt cytosolic proteins as a negative source —\n"
            "that source was evaluated and rejected for this project.\n"
            "Negatives are built by process_negatives.py; run that script\n"
            "and point amp_neg_path at its amp_neg_clf.csv output."
        )

    logger.info("Loading AMP negatives from %s", neg_path)
    neg_df = _read_amp_negatives(neg_path)
    neg_df = _clean_sequences(neg_df, AMP_SEQ_COL, min_len, max_len)
    neg_df = neg_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    neg_df['label'] = 0
    logger.info("AMP negatives after cleaning: %d", len(neg_df))

    # Remove any negatives that appear in positives (sequence overlap)
    pos_seqs = set(pos_df['sequence'])
    neg_df = neg_df[~neg_df['sequence'].isin(pos_seqs)]
    logger.info(
        "AMP negatives after removing positives overlap: %d", len(neg_df)
    )

    dataset = pd.concat([pos_df, neg_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = dataset['label'].sum()
    n_neg = (dataset['label'] == 0).sum()
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    logger.info(
        "AMP dataset: %d positive, %d negative  (ratio %.2f)",
        n_pos, n_neg, ratio,
    )
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            "Class imbalance ratio %.2f is severe. Ensure classifiers use "
            "class_weight='balanced' or apply undersampling.", ratio
        )

    return dataset


def build_cpp_dataset(
    cpp_path: str,
    neg_path: Optional[str],
    min_len: int = 5,
    max_len: int = 30,
) -> pd.DataFrame:
    """
    Build a labelled CPP dataset.

    All sequences in cpp_pos_clf.csv are experimentally validated CPPs
    (label=1). A separate file of non-CPP sequences is required for the
    negative class: 42 benchmark non-CPPs + 175 UniProt chain-derived
    sequences (19% experimental overall), built by process_negatives.py.
    CPPsite 2.0's non-CPP set was evaluated and rejected for this project
    — it is randomly generated from Swiss-Prot, not experimentally
    validated, and must not be used as a substitute.
    """
    logger.info("Loading CPP positives from %s", cpp_path)
    df = _read_cpp_dataset(cpp_path)
    df = df.rename(columns={AMP_SEQ_COL: 'sequence'})
    df = _clean_sequences(df, 'sequence', min_len, max_len)
    df = df[['sequence']].copy()
    df['label'] = 1
    logger.info("CPP positives after cleaning: %d", len(df))

    if neg_path is None or not Path(neg_path).exists():
        raise FileNotFoundError(
            "No CPP negative dataset found at path: "
            f"'{neg_path}'.\n"
            "CPP negatives must be the process_negatives.py output\n"
            "(42 benchmark non-CPPs + 175 UniProt chain-derived, 19%\n"
            "experimental). Do NOT substitute CPPsite 2.0's non-CPP set —\n"
            "that source was evaluated and rejected: it is randomly\n"
            "generated from Swiss-Prot, not experimentally validated.\n"
            "Run process_negatives.py and point cpp_neg_path at its\n"
            "cpp_neg_clf.csv output."
        )

    logger.info("Loading CPP negatives from %s", neg_path)
    neg_df = _read_amp_negatives(neg_path)   # reuse reader (expects 'Sequence')
    neg_df = _clean_sequences(neg_df, AMP_SEQ_COL, min_len, max_len)
    neg_df = neg_df[[AMP_SEQ_COL]].rename(columns={AMP_SEQ_COL: 'sequence'})
    neg_df['label'] = 0
    logger.info("CPP negatives after cleaning: %d", len(neg_df))

    # Remove overlap
    pos_seqs = set(df['sequence'])
    neg_df = neg_df[~neg_df['sequence'].isin(pos_seqs)]
    logger.info("CPP negatives after overlap removal: %d", len(neg_df))

    dataset = pd.concat([df, neg_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    n_pos = dataset['label'].sum()
    n_neg = (dataset['label'] == 0).sum()
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    logger.info("CPP dataset: %d positive, %d negative  (ratio %.2f)",
                n_pos, n_neg, ratio)
    if ratio > 5 or ratio < 0.2:
        logger.warning(
            "Class imbalance ratio %.2f is severe. Classifiers will use "
            "class_weight='balanced'.", ratio
        )
    return dataset


# =============================================================================
# Stratified 3-way split
# =============================================================================

def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    CRITICAL FIX:
    Original pipeline had NO test set — cross-validation was the only
    evaluation. Calibration was performed on training data, not a held-out
    validation set. This function creates a proper 3-way split:
      - Train: used for model fitting and CV
      - Validation: used for calibration and hyperparameter selection
      - Test: held-out, used ONLY for final metric reporting

    The test set is never touched during training or calibration.
    """
    y = df['label'].values

    # First split off the test set
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=y, random_state=seed
    )
    # Then split validation from the remaining train+val
    val_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val['label'].values,
        random_state=seed,
    )

    logger.info(
        "Split → train: %d  val: %d  test: %d",
        len(train), len(val), len(test),
    )

    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        n_pos = split_df['label'].sum()
        n_neg = (split_df['label'] == 0).sum()
        logger.info("  %s: %d pos / %d neg", split_name, n_pos, n_neg)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
