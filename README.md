# Hybrid Therapeutic Peptide Generator — CDSPO

> **C**alibrated **D**ual-**S**core **P**eptide **O**ptimisation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20594473.svg)](https://doi.org/10.5281/zenodo.20594473)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MethodsX](https://img.shields.io/badge/Published-MethodsX-green.svg)](#citation)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org)

A generative AI pipeline for designing **hybrid therapeutic peptides** that simultaneously carry **antimicrobial (AMP)** and **cell-penetrating (CPP)** activity — candidate scaffolds for treating drug-resistant intracellular infections.

> **Companion dataset:** Upadhyay A. *CVAE training data and generated peptide/enzyme candidates for AMP, CPP and cellulase design.* Zenodo, 2026.
> DOI: [10.5281/zenodo.20594473](https://doi.org/10.5281/zenodo.20594473)

---

## Why this repository exists

Classical AMP discovery targets extracellular bacteria. For **intracellular pathogens** (*Mycobacterium tuberculosis*, *Listeria monocytogenes*, *Salmonella* spp.) the antimicrobial agent must first penetrate the host cell membrane before it can act. Peptides that are **both** AMP-positive and CPP-positive — here called *hybrid therapeutic peptides* — can, in principle, self-deliver to the intracellular compartment and bypass classical efflux-mediated resistance.

Such dual-active sequences are rare in nature and hard to obtain by mutagenesis because the two activities place partly correlated, partly competing physicochemical demands on the sequence. This pipeline addresses that problem through calibrated generative AI.

### What is new in this approach

| Feature | Earlier approaches | **This pipeline (CDSPO)** |
|---|---|---|
| Negative class | Synthetically shuffled positives | Experimentally validated inactives from six independent databases |
| Sequence representation | k-mer + SVD | **ESM-2** contextual embeddings + 12 physicochemical descriptors |
| Scoring | Single classifier, uncalibrated | **Calibrated ensemble** (GBM + MLP + RF) per task with held-out isotonic calibration |
| Generation objective | Maximise one property | **Geometric-mean dual score** √(p_AMP × p_CPP) — requires both activities simultaneously |
| Diversity control | None or simple deduplication | Levenshtein edit-distance filter + K-means physicochemical clustering (k = 20, top-4 per cluster) |
| Output | Unconstrained candidate list | **80 candidates** spanning 20 physicochemical clusters, 11–45 residues |

---

## Key numbers

| Quantity | Value |
|---|---|
| AMP training positives (ADP v6, disulfide-bond entries) | n = 1,655 |
| CPP training positives (CellPPD / Raghavendra) | n = 709 |
| AMP negatives (DBAASP / LAMP2 / PepBDB / UniProt) | 3× positives |
| CPP negatives (CPPsite 2.0/3.0 / MLCPP2 / UniProt) | 3× positives |
| CVAE latent dimension | 64 |
| Score-guided generation rounds | 5 |
| Final candidate set | **80 peptides / 20 clusters** |
| Candidate length range | 11 – 45 residues |
| Mean combined score | 0.9986 (range 0.989 – 1.000) |
| Mean net charge at pH 7.4 | +6.4 |
| Mean pI | 11.4 |

---

## Repository structure

```
CDSPO/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── config/
│   ├── default.yaml          # All hyperparameters, thresholds, paths
│   ├── amp_broad.yaml        # Broader AMP positive set (no disulfide filter)
│   ├── diversity.yaml        # K-means k and per-cluster top-k settings
│   └── toxicity_filter.yaml  # Downstream toxicity / haemolysis screens
│
├── data/
│   ├── README.md             # Explains Zenodo deposit; no raw data committed here
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_curation.py      # Dataset assembly and cross-source consolidation
│   ├── representation.py     # ESM-2 embeddings + 12 physicochemical descriptors
│   ├── classifiers.py        # GBM + MLP + RF ensembles, calibration, evaluation
│   ├── cvae.py               # CVAE architecture (encoder, decoder, masked loss)
│   ├── generation.py         # Score-guided sampling loop (5 rounds)
│   ├── filters.py            # Multi-criterion biological filter
│   ├── diversity.py          # Edit-distance filter + K-means selection
│   └── utils.py              # Shared helpers
│
├── scripts/
│   ├── train_classifiers.py  # Train + calibrate AMP and CPP ensembles
│   ├── train_cvae.py         # Train the conditional VAE generator
│   ├── generate.py           # End-to-end generation pipeline
│   ├── novelty_analysis.py   # Nearest-neighbour identity vs training corpus
│   ├── ablation_score.py     # Geometric vs arithmetic vs min scoring
│   └── ablation_filter.py    # Per-filter survival counts
│
├── models/
│   ├── README.md             # Download instructions (Zenodo link)
│   └── .gitkeep              # Model weights NOT committed (too large)
│
├── outputs/
│   └── .gitkeep
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_classifier_evaluation.ipynb
    └── 03_candidate_analysis.ipynb
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/computational-genomics-lab/hybrid-peptide-CDSPO.git
cd hybrid-peptide-CDSPO
```

### 2. Create the conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate cdspo
```

### 3. Or install with pip

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Version | Role |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| PyTorch | ≥ 2.0 | CVAE training and inference |
| fair-esm | ≥ 2.0 | ESM-2 contextual embeddings |
| scikit-learn | ≥ 1.4 | Classifiers, calibration, clustering |
| BioPython | ≥ 1.81 | Physicochemical descriptor computation |
| NumPy | 1.26 | Array operations |
| pandas | 2.0 | Data handling |
| python-Levenshtein | any | Edit-distance diversity filter |

---

## Data

All training data and generated candidate outputs are deposited at Zenodo under **CC-BY 4.0**:

> **Upadhyay A.** *CVAE training data and generated peptide/enzyme candidates for AMP, CPP and cellulase design.* Zenodo (2026).
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20594473.svg)](https://doi.org/10.5281/zenodo.20594473)

### Files relevant to this pipeline

| File | Size | Contents |
|---|---|---|
| `AMP_positive.xlsx` | 217 kB | 1,655 experimentally validated AMP sequences (ADP v6, disulfide-bond entries) |
| `CPP_positive.csv` | 77 kB | 709 experimentally validated CPP sequences (CellPPD / Raghavendra) |
| `generated_hybridPeptides_candidates.csv` | 10 kB | 80 final hybrid AMP/CPP candidate sequences with predicted scores |

> Negative-class data (DBAASP, LAMP2, PepBDB, CPPsite 2.0/3.0, MLCPP2, UniProt) must be downloaded directly from the respective source databases because their licences do not permit bulk redistribution. See `data/README.md` for exact query strings and download instructions.

### Download the Zenodo files

```bash
# Download all files from the Zenodo record
wget https://zenodo.org/records/20594473/files/AMP_positive.xlsx
wget https://zenodo.org/records/20594473/files/CPP_positive.csv
wget https://zenodo.org/records/20594473/files/generated_hybridPeptides_candidates.csv

# Place them in data/
mv AMP_positive.xlsx CPP_positive.csv generated_hybridPeptides_candidates.csv data/
```

---

## Pipeline walkthrough

The pipeline runs in three sequential stages. All hyperparameters are externalised in `config/default.yaml` and can be overridden with `--set key=value`.

```
Stage 1            Stage 2                Stage 3
─────────────      ───────────────────    ──────────────────────────────
Data curation  →   Train classifiers  →   Train CVAE
                   + CVAE             →   Score-guided generation
                                      →   Biological filter
                                      →   Diversity selection
                                      →   Final 80 candidates
```

### Stage 1 — Assemble and curate datasets

```bash
python scripts/train_classifiers.py --config config/default.yaml --mode data_only
```

What this does:
- Downloads and consolidates AMP/CPP positives and negatives
- Applies length filters (5–60 aa for AMP, 5–30 aa for CPP)
- Deduplicates within and across sources with source-priority weighting
- Caps negative pool at 3× positives
- Computes and caches ESM-2 embeddings (requires GPU; approximately 20 min on A100)
- Computes 12 physicochemical descriptors per sequence

### Stage 2 — Train classifiers

```bash
python scripts/train_classifiers.py --config config/default.yaml
```

This trains two calibrated classifier ensembles (one for AMP, one for CPP):

1. **Gradient Boosting Machine** — 300 trees, lr 0.07, max_depth 4, subsample 0.8
2. **Multilayer Perceptron** — 256 → 128 → 64 units, ReLU, Adam 1×10⁻⁴
3. **Random Forest** — 300 trees, min_samples_leaf = 3

Calibration is applied per classifier using `CalibratedClassifierCV(method='isotonic', cv='prefit')` on a held-out 15 % validation set. All three classifiers are averaged to form the ensemble probability.

Output metrics printed to terminal and saved to `outputs/classifier_metrics.json`:
- Sealed-test ROC-AUC, PR-AUC, F1, MCC, Brier score, ECE, MCE
- Reliability diagrams saved as `outputs/reliability_AMP.png` and `outputs/reliability_CPP.png`

### Stage 3 — Train CVAE and generate candidates

```bash
# Train the conditional VAE
python scripts/train_cvae.py --config config/default.yaml

# Run end-to-end generation (score-guided optimisation → filter → diversity → output)
python scripts/generate.py --config config/default.yaml
```

**CVAE architecture:**
- Encoder: 1D-CNN with parallel kernels (width 3, 5, 7) → mask-aware global average pool → μ, log σ² (latent dim = 64)
- Decoder: position-wise MLP with length-conditioning embedding → per-position softmax (B × L × 20)
- Loss: masked categorical cross-entropy + β(t) · KL, with linear KL annealing (β_warmup = 60 epochs, total = 150 epochs)

**Score-guided generation (5 rounds):**

```
For each round:
  1. Draw 3,000 latent vectors from N(0, I); decode to sequences
  2. Score each with AMP and CPP calibrated ensembles
  3. Combined score c = √(p_AMP × p_CPP)
  4. Softmax-weighted seed selection (temperature T = 0.3)
  5. Apply 1–3 random substitutions per seed
  6. Merge with previous round; deduplicate; re-rank
```

**Biological filter (hard rules, applied after generation):**

| Criterion | Threshold | Biological rationale |
|---|---|---|
| Length | 8 – 50 residues | Synthetic accessibility |
| Net charge at pH 7.4 | ≥ +1.0 | Electrostatic attachment to anionic bacterial membranes |
| GRAVY | < 2.5 | Aggregation prevention |
| Instability index | < 60 | Solution stability |
| Hydrophobic moment | ≥ 0.0 | Helical amphipathicity for membrane disruption |
| Aggregation propensity | < 5 (max consec. hydrophobic run) | β-aggregation prevention |
| p_AMP | ≥ 0.50 | Both activities above calibrated threshold |
| p_CPP | ≥ 0.50 | Both activities above calibrated threshold |

**Diversity selection:**
1. Greedy Levenshtein edit-distance filter: identity threshold 80 %
2. K-means clustering (k = 20) in scaled physicochemical descriptor space
3. Top-4 candidates per cluster → **80 final candidates**

---

## Running the full pipeline in one command

```bash
python scripts/generate.py \
    --config config/default.yaml \
    --set data.amp_positive=data/AMP_positive.xlsx \
    --set data.cpp_positive=data/CPP_positive.csv \
    --set output.dir=outputs/run_001 \
    --seed 42
```

A timestamped run directory is created under `outputs/run_001/` containing:

```
outputs/run_001/
├── config_used.yaml          # Exact config for reproducibility
├── classifier_metrics.json   # Sealed-test AUC, Brier, ECE, MCE
├── reliability_AMP.png       # Reliability diagram — AMP ensemble
├── reliability_CPP.png       # Reliability diagram — CPP ensemble
├── generation_log.csv        # Per-round pool statistics
├── candidates_all.csv        # All post-filter candidates
├── candidates_final.csv      # 80 diversity-selected candidates (main output)
├── candidates_final.fasta    # FASTA format of final 80 sequences
├── dump_pool_round{1-5}.csv  # Full pool after each optimisation round
└── run.log                   # Full execution log with SHA-256 input hashes
```

---

## Output format

`candidates_final.csv` columns:

| Column | Description |
|---|---|
| `rank` | Global rank by combined score |
| `sequence` | Amino acid sequence (canonical alphabet) |
| `amp_score` | Calibrated AMP ensemble probability |
| `cpp_score` | Calibrated CPP ensemble probability |
| `combined_score` | Geometric mean √(amp × cpp) |
| `cluster` | K-means cluster ID (0–19) |
| `length` | Sequence length (residues) |
| `molecular_weight` | Da |
| `pI` | Isoelectric point (Henderson–Hasselbalch) |
| `net_charge` | Net charge at pH 7.4 |
| `instability_index` | Guruprasad instability index |
| `GRAVY` | Kyte–Doolittle hydropathy |
| `hydrophobic_moment` | Eisenberg (window = 11, 100°/residue) |
| `aggregation_propensity` | Max consecutive hydrophobic run |
| `aromaticity` | F+W+Y fraction |
| `aliphatic_index` | Relative volume occupied by aliphatic side chains |
| `boman_index` | Protein–protein interaction potential |
| `ww_hydrophobicity` | Wimley–White interfacial hydrophobicity |

---

## Configuration

All parameters live in `config/default.yaml`. Key sections:

```yaml
data:
  amp_positive: data/AMP_positive.xlsx
  cpp_positive: data/CPP_positive.csv
  amp_neg_ratio: 3            # negatives = ratio × positives
  amp_length_min: 5
  amp_length_max: 60
  cpp_length_max: 30

representation:
  esm2_model: esm2_t12_35M_UR50D
  esm2_freeze: true           # DO NOT fine-tune ESM-2 weights
  n_descriptors: 12

classifiers:
  n_trees: 300
  cv_folds: 5
  calibration: isotonic

cvae:
  latent_dim: 64
  epochs: 150
  kl_warmup_epochs: 60
  beta: 1.0
  batch_size: 64

generation:
  n_initial: 3000
  n_rounds: 5
  temperature: 0.3
  mutations_per_seed: [1, 3]

filter:
  length_min: 8
  length_max: 50
  charge_min: 1.0
  gravy_max: 2.5
  instability_max: 60
  hm_min: 0.0
  agg_max: 5
  amp_prob_min: 0.50
  cpp_prob_min: 0.50

diversity:
  identity_threshold: 0.80    # Levenshtein edit-distance filter
  n_clusters: 20
  top_k_per_cluster: 4

output:
  dir: outputs/
  seed: 42
```

Override any parameter at runtime:

```bash
python scripts/generate.py --set diversity.n_clusters=10 --set filter.instability_max=40
```

---

## Downstream validation (mandatory before any biological claims)

The 80 candidates produced by this pipeline are **computational prioritisations only**. Before any therapeutic conclusions can be drawn, the following wet-lab assays are required:

1. **Minimum inhibitory concentration (MIC)** — Broth microdilution against standard Gram-positive (*S. aureus* ATCC 29213) and Gram-negative (*E. coli* ATCC 25922) panels
2. **Haemolysis assay** — Human erythrocyte lysis at 2× and 10× MIC
3. **Cell-uptake / CPP activity** — Fluorescence confocal microscopy of FITC-labelled candidates in HeLa or RAW264.7 macrophage cells
4. **Computational toxicity triage (minimal pre-synthesis screen)** — ToxinPred 3.0, HemoPI 2.0, AlgPred 2.0 (configs/toxicity_filter.yaml)

---

## Notebooks

Interactive Jupyter notebooks are provided for exploration:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Visualise AMP/CPP dataset composition, source balance, length/charge distributions |
| `02_classifier_evaluation.ipynb` | Reliability diagrams, calibration curves, ROC/PR plots per classifier and ensemble |
| `03_candidate_analysis.ipynb` | Score landscape, physicochemical profile, cluster diversity, novelty analysis of the 80 candidates |

```bash
jupyter notebook notebooks/
```

---

## Reproducibility

Every run is fully reproducible from its `outputs/run_*/config_used.yaml` and the SHA-256 hashes logged in `run.log`. To re-run an existing experiment exactly:

```bash
python scripts/generate.py --config outputs/run_001/config_used.yaml
```

---

## How to cite

If you use this pipeline or the companion dataset, please cite:

**Dataset:**
```bibtex
@dataset{upadhyay_2026_20594473,
  author       = {Upadhyay, Aditya},
  title        = {CVAE training data and generated peptide/enzyme
                  candidates for AMP, CPP and cellulase design},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.20594473},
  url          = {https://doi.org/10.5281/zenodo.20594473}
}
```

**Method article (MethodsX — in preparation):**
```bibtex
@article{upadhyay_2026_cdspo,
  author  = {Upadhyay, Aditya and [Co-authors]},
  title   = {A calibrated dual-score generative pipeline for
             prioritising peptides predicted to combine antimicrobial
             and cell-penetrating activity},
  journal = {MethodsX},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

**Companion thesis chapter:**
Upadhyay A. *et al.* Chapter 5 in *Big data analytics using contemporary algorithms and data warehousing.* [Publisher], 2026.

---

## Known limitations

- The 80 final candidates are **in silico prioritisations only** — no wet-lab activity has been confirmed for the generated sequences.
- The AMP negative set (43 sequences after curation) is substantially smaller than the AMP positive set, widening confidence intervals on AMP probability estimates for novel sequences.
- ESM-2 embeddings were computed with model weights frozen; fine-tuning on AMPs/CPPs may improve representation quality but risks overfitting on the small corpus.
- The pipeline does not include structural prediction (AlphaFold2, ESMFold); adding a structural sanity check is recommended for the longer (≥ 35 residue) candidates.
- The disulfide-bond restriction on the AMP positive set biases recovered scaffolds toward cysteine-rich, defensin-class structures. For helical AMP scaffolds, use `config/amp_broad.yaml`.

---

## Affiliation

Developed at the **Structural Topology and Computational Biology Laboratory (STLAB)**
CSIR – Indian Institute of Chemical Biology, Kolkata, India
AcSIR — Academy of Scientific and Innovative Research

---

## License

MIT License — see [LICENSE](LICENSE) for details.

The companion **dataset** (Zenodo record 20594473) is released under **Creative Commons Attribution 4.0 International (CC-BY 4.0)**.
