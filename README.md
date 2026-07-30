# Evidence-tiered negative curation and cross-classifier specificity control

> Demonstrated on **CDSPO** — **C**alibrated **D**ual-**S**core **P**eptide **O**ptimisation

[![DOI](https://img.shields.io/badge/Zenodo-DOI%20to%20be%20minted-lightgrey.svg)](#data)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org)

This repository implements two transferable methods for peptide bioactivity modelling, together with the generative pipeline used to demonstrate them:

1. **Evidence-tiered negative-set construction** — building negative classes from *positive evidence of inactivity* rather than from the absence of a positive annotation.
2. **A cross-classifier specificity control** — a post-training diagnostic that distinguishes a classifier which learned the intended activity from one which learned a property of the sampling frame.

The demonstration application is **CDSPO**, a conditional-VAE pipeline generating peptides intended to carry both antimicrobial (AMP) and cell-penetrating (CPP) activity — candidate scaffolds for drug-resistant *intracellular* infections.

> **Read this first.** Applying the specificity control to CDSPO showed that **the AMP term in the dual objective is non-binding**: random peptides of matched length score as highly as real antimicrobial peptides under the AMP classifier. This is reported openly below and in the manuscript. It is the finding the diagnostic was built to catch, and it changes how the candidate list should be read.

---

## Why this repository exists

Classical AMP discovery targets extracellular bacteria. For intracellular pathogens (*Mycobacterium tuberculosis*, *Listeria monocytogenes*, *Salmonella* spp.) an antimicrobial must first cross the host cell membrane. Peptides that are both AMP- and CPP-active can in principle self-deliver to the intracellular compartment.

Building supervised models for either activity requires a negative class, and repositories of *experimentally inactive* peptides are far smaller than repositories of active ones. The usual workarounds — shuffling positives, or sampling proteins carrying no positive annotation — define the negative class by what it lacks. Any systematic difference between the two sampling frames (length, composition, subcellular provenance) then becomes available as a shortcut, and the reported metrics measure frame separability rather than biology.

This is not hypothetical. An earlier configuration of this pipeline trained the AMP classifier against 43 UniProt cytoplasmic proteins selected by the absence of an antimicrobial keyword. It reported **sealed-test ROC-AUC 0.996 and PR-AUC 1.000** — on a test partition holding roughly **six negatives**. Random peptides scored 0.862 under that model. What it had learned was short-cationic-peptide versus long-cytoplasmic-protein.

### What this repository contributes

| | Common practice | **This repository** |
|---|---|---|
| AMP negatives | Shuffled positives, or proteins lacking an AMP keyword | **DBAASP records with min MIC ≥ 256 µg/mL against ≥ 3 species** — quantitative inactivity, 335 sequences |
| CPP negatives | CPPsite bundled "non-CPP" files | **Two evidence tiers**: 42 experimentally tested non-penetrating peptides + 175 UniProt mature chains, with three common sources excluded for documented reasons |
| Unit handling | Implicit | Explicit µM → µg/mL conversion per peptide (sequence MW, or RDKit from SMILES for modified peptides) with an **abort-on-breakage self-test** |
| Classifier validation | ROC/PR on a held-out split | **Five-population cross-classifier control** — each classifier scored against *both* negative sets plus a length-matched random baseline |
| Novelty reporting | "% novel" | Distance distribution **against the full reference pool**, with the subsampling artefact documented |
| Claims | "our method improves…" | Design rationale stated; the one available comparison is labelled **confounded**, not presented as an ablation |

---

## Key numbers

Seed-42 run; these match the manuscript.

### Datasets

| Quantity | Value |
|---|---|
| AMP positives (DRAMP 4.0 + DBAASP + ADP v6 + CAMPR4) | ~35,000 dedup → 15,283 at 5–50 aa → **9,317** after HemoPI2 |
| AMP positive length | mean 17.2 ± 8.2, median 16, IQR 11–20 |
| AMP negatives (DBAASP, quantitative inactivity) | 345 at ≥ 3 species → 342 after length → **335** canonical |
| AMP classifier split | 1,005 pos / 335 neg (3:1) |
| AMP generator pool | **9,205** (9,316 minus 111 tier-1 inactives) |
| CPP positives (CPPsite 1–3 + CellPPD) | 1,596 → 856 dedup → 673 HemoPI2 → 660 at 5–30 aa → **652** |
| CPP negatives tier 1 (tested non-penetrating) | 43 unique → **42** after length |
| CPP negatives tier 2 (UniProt mature chains) | 3,238 → 1,897 unique → **1,251** after redundancy reduction |
| CPP classifier split | 651 pos / 217 neg (42 tier-1 + 175 tier-2) |
| Experimental fraction of CPP negatives | **19 %** |

### Classifier performance (sealed test, seed 42)

| Task | ROC-AUC | PR-AUC | Brier | CV ROC-AUC (GBM / MLP / RF) |
|---|---|---|---|---|
| AMP | 0.785 | 0.900 | 0.136 | 0.754 / 0.788 / 0.782 |
| CPP | 0.908 | 0.963 | 0.105 | 0.854 / 0.854 / 0.857 |

These are **lower than the earlier configuration and that is the point** — the task is better posed. See [The AMP result](#the-amp-result--read-this-before-using-the-candidates).

### Candidates

| Quantity | Value |
|---|---|
| Final candidate set | **80 peptides / 20 clusters** (4 per cluster) |
| Length | 9 – 28 aa (mean 18.0 ± 5.7) |
| Calibrated AMP score | mean 0.867 ± 0.031 |
| Calibrated CPP score | mean 0.970 ± 0.021 |
| Combined score | mean 0.917 (0.883 – 0.967) |
| Net charge @ pH 7.4 | +1.80 to +15.79 (mean +6.34) |
| pI | 10.23 – 13.11 (mean 11.59) |
| Instability index | 6.77 – 49.49 (mean 18.80) |
| Hydrophobic moment | 0.047 – 0.394 (mean 0.241) |
| Novelty vs full 9,205 pool | mean distance 0.476, median 0.500, min 0.091, **0 % memorised**, 38.75 % > 0.5 |

---

## The AMP result — read this before using the candidates

The five-population specificity control produces this matrix (calibrated scale, seed 42; `score_comparison.csv`):

| Population | n | AMP | CPP | Combined |
|---|---|---|---|---|
| Generated candidates | 500 | 0.763 | 0.696 | 0.715 |
| Training positives | 500 | 0.753 | 0.617 | 0.668 |
| AMP negatives *(own)* | 335 | **0.366** | 0.653 | 0.443 |
| CPP negatives *(own)* | 217 | 0.744 | **0.235** | 0.349 |
| Random baseline | 500 | 0.763 | 0.468 | 0.583 |

Two readings, pointing in different directions.

**Favourable.** Each classifier rejects only *its own* negatives and scores the other task's negatives highly (AMP model: 0.366 own vs 0.744 CPP-negatives; CPP model: 0.235 own vs 0.653 AMP-negatives). The two classifiers therefore encode different biology, not a shared "atypical peptide" axis. Positive-to-own-negative separation is closely matched: **AMP +0.387, CPP +0.382**.

**Unfavourable.** Random peptides score **0.763** under the AMP classifier — indistinguishable from generated candidates (0.763) and *above* training positives (0.753). A high AMP score means "unlike the experimentally inactive peptides this model was trained on", **not** "antimicrobial". Random sequences are out-of-distribution for both trained classes and are not rejected.

**Consequences for anyone using this repository:**

- The AMP term exerts **effectively no selective pressure** during score-guided optimisation. Over the explored region, `combined = √(p_AMP · p_CPP)` is close to a monotone function of `p_CPP` alone.
- AMP-likeness in the final candidate set comes from the **CVAE prior** (trained on 9,205 validated AMPs) and the cationicity/amphipathicity filters — not from the AMP classifier.
- Within the final 80, isotonic calibration leaves only **9 distinct CPP values and 25 AMP values**, so rank order is not meaningful at fine granularity. **Select by cluster membership and physicochemical diversity, not by rank.**

The raw-scale matrix (`score_comparison_raw.csv`) shows the same structure with wider spread: generated 0.665 / 0.676, positives 0.637 / 0.595, AMP-negatives 0.318 / 0.636, CPP-negatives 0.623 / 0.264, random 0.647 / 0.440.

---

## Repository structure

```
.
├── main.py                          # Entry point — full pipeline
├── run_calibration_diagnostic.py    # Standalone: raw vs calibrated score spread
├── config.yaml                      # All paths, hyperparameters, thresholds
│
├── data/
│   └── loader.py                    # build_amp_dataset, build_cpp_dataset,
│                                    #   load_generator_positives
├── features/
│   ├── esm_embeddings.py            # ESM-2 (frozen), mean-pooled
│   ├── physicochemical.py           # 12 descriptors
│   ├── transformers.py              # sklearn-compatible wrappers
│   └── feature_factory.py           # Assembles the final feature vector
├── models/
│   ├── classifier.py                # GBM + MLP + RF, isotonic calibration,
│   │                                #   stratified_split, evaluate_on_test
│   ├── cvae_torch.py                # CVAE generator (used in the manuscript)
│   └── cvae_numpy.py                # Reference NumPy implementation
├── generation/
│   └── pipeline.py                  # Score-guided optimisation, levenshtein
├── filters/
│   └── biological.py                # Charge, GRAVY, instability, moment, aggregation
├── diversity/
│   └── cluster.py                   # Edit-distance filter + K-means
├── training/
│   ├── train.py                     # Classifier + CVAE training orchestration
│   └── evaluate.py                  # novelty_report, compare_score_distributions,
│                                    #   generate_random_peptides
├── diag/
│   └── check_calibration_spread.py  # diagnose() — library, no __main__
└── utils/
    └── logging_utils.py
```

> **Not yet in this repository:** the curation scripts that *produce* the six input CSVs — DBAASP download and inactivity parsing, UniProt mature-chain extraction, tier assembly, conflict resolution, length matching. These implement the protocol the manuscript describes as its contribution, and **must be added before the Data Availability statement is accurate.** See [Outstanding](#outstanding).

---

## Installation

```bash
git clone https://github.com/computational-genomics-lab/hybrid-peptide-CDSPO.git
cd hybrid-peptide-CDSPO

conda create -n amp_cpp python=3.10 -y
conda activate amp_cpp
pip install -r requirements.txt

# CPU-only torch (optional, strongly recommended if you have no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Core dependencies**

| Package | Version | Role |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| PyTorch | ≥ 2.0 | CVAE training and inference |
| fair-esm | ≥ 2.0 | ESM-2 contextual embeddings |
| scikit-learn | ≥ 1.4 | Classifiers, isotonic calibration, K-means |
| BioPython | ≥ 1.81 | Physicochemical descriptors |
| RDKit | any | MW from SMILES for modified peptides (curation stage) |
| NumPy | 1.26 | Array operations |
| pandas | 2.0 | Data handling |
| PyYAML | any | Configuration |

---

## Data

Curated datasets and generated outputs will be deposited at Zenodo under CC-BY 4.0.

> **DOI: `[to be minted]`** — a new deposit is required. Any earlier Zenodo record for this project holds the **superseded** pre-recuration datasets and must not be cited for this work.

### Required input layout

Six CSVs, each with a `sequence` header, at the paths named in `config.yaml`:

```
output/
└── seed42/
    ├── amp_pos_clf.csv               (1,005 rows)  AMP classifier positives
    ├── amp_neg_clf.csv               (  335 rows)  AMP classifier negatives
    ├── cpp_pos_clf.csv               (  651 rows)  CPP classifier positives
    ├── cpp_neg_clf.csv               (  217 rows)  CPP classifier negatives
    ├── amp_generator_positives.csv   (9,205 rows)  CVAE training pool (AMP)
    └── cpp_generator_positives.csv   (  652 rows)  CVAE training pool (CPP)
```

> ⚠️ **Path collision.** Inputs live under `output/` (singular); results are written to `outputs/` (plural). These are different directories. Check which you are looking at before concluding a run failed.

The classifier and the generator deliberately train on **different data**: the classifier uses a 3:1 negative-capped split, the generator uses the full positive pool. At the natural pool ratio (~27:1) the AMP classifier stops discriminating; 3:1 is the regime in which the CPP classifier operates.

Negative-class source data (DBAASP, UniProt, CPPsite, CellPPD benchmarks) must be obtained from the respective providers — their licences do not permit bulk redistribution.

---

## Running the pipeline

### 1. Sanity-check the loaders first

Do not commit to a full run with a broken loader.

```bash
python -c "
import sys; sys.path.insert(0, '.')
from data.loader import build_amp_dataset, build_cpp_dataset, load_generator_positives
import yaml

cfg = yaml.safe_load(open('config.yaml'))['data']

amp = build_amp_dataset(cfg['amp_pos_path'], cfg['amp_neg_path'],
                        cfg['amp_min_len'], cfg['amp_max_len'])
cpp = build_cpp_dataset(cfg['cpp_path'], cfg['cpp_neg_path'],
                        cfg['cpp_min_len'], cfg['cpp_max_len'])
amp_gen = load_generator_positives(cfg['amp_gen_path'], cfg['amp_min_len'], cfg['amp_max_len'])
cpp_gen = load_generator_positives(cfg['cpp_gen_path'], cfg['cpp_min_len'], cfg['cpp_max_len'])

print('AMP classifier:', amp['label'].value_counts().to_dict())
print('CPP classifier:', cpp['label'].value_counts().to_dict())
print('AMP generator pool:', len(amp_gen))
print('CPP generator pool:', len(cpp_gen))
"
```

Expect approximately the row counts above — slightly lower after length and canonical-residue filtering.

### 2. Full run

```bash
python main.py --config config.yaml
```

### 3. Faster first run

Confirm everything wires up before committing to 150 CVAE epochs:

```bash
python main.py --config config.yaml \
  --set generator.epochs=30 \
  --set generator.kl_warmup_epochs=15
```

### 4. A different dataset seed

```bash
python main.py --config config.yaml \
  --set data.amp_pos_path=output/seed43/amp_pos_clf.csv \
  --set data.amp_neg_path=output/seed43/amp_neg_clf.csv \
  --set data.cpp_path=output/seed43/cpp_pos_clf.csv \
  --set data.cpp_neg_path=output/seed43/cpp_neg_clf.csv \
  --set data.amp_gen_path=output/seed43/amp_generator_positives.csv \
  --set data.cpp_gen_path=output/seed43/cpp_generator_positives.csv
```

> `--set seed=43` changes **only** the model seed (shuffling, CV folds, torch init). It does **not** select a different dataset — override the six `data.*` paths as above. The model seed is held at 42 across dataset draws so that variance is attributable to the dataset, not the model.

---

## Reproducing each published artefact

| Artefact | Emitted by | Called from | How to regenerate |
|---|---|---|---|
| `amp_metrics.json`, `cpp_metrics.json` | `models/classifier.py` → `evaluate_on_test()` | `main.py` (classifier phases) | Full pipeline run |
| `score_comparison.csv` (calibrated) | `training/evaluate.py` → `compare_score_distributions()` | `main.py:387` | Full pipeline run |
| `score_comparison_raw.csv` (raw) | same function, second call | `main.py:407` | Full pipeline run |
| `novelty_report.json` | `training/evaluate.py:177` → `novelty_report()` | `main.py:419` (Phase 8) | Full pipeline run |
| `candidates.csv`, `report.txt` | `main.py` final phase | — | Full pipeline run |
| Calibration spread, **test** set | `diag/check_calibration_spread.py` → `diagnose()` | `run_calibration_diagnostic.py` | Standalone, minutes |
| Calibration spread, **validation** set | `models/classifier.py` auto-logging | every `main.py` run | Read the metrics JSON |

**There is no standalone script for novelty or score-comparison.** Both need pipeline-internal state — the final candidate set, the combined generator pool, and the `predict_*_cal` / `predict_*_raw` callables — which only exist mid-run. Regenerating either means a full `main.py` invocation.

### Calibration granularity — two distinct measurements

These are different splits and are not interchangeable:

**Validation set**, auto-logged to the metrics JSON on every run as `val_raw_unique_scores` / `val_calibrated_unique_scores`:

| Task | Raw unique values | Calibrated unique values |
|---|---|---|
| AMP | 201 | **73** |
| CPP | 131 | **42** |

Isotonic calibration compresses the AMP dynamic range (positives std 0.157 raw → 0.097 calibrated, a 38 % loss) while leaving CPP largely intact (0.191 → 0.187). This is why **raw scores steer generation and calibrated scores are reported** — calibrated scores inside the optimisation loop would blunt the search.

**Test set**, printed to stdout only:

```bash
python run_calibration_diagnostic.py --config config.yaml
```

This retrains both classifiers exactly as `main.py` does, then reports the raw-versus-calibrated spread on each held-out **test** split. It skips the CVAE and generation phases, so it completes in minutes. Look for the line `overall unique probability values in test set: <cal> / <raw>`. Test-set figures differ slightly from the validation figures above; quote whichever split you actually used and say which.

> `diag/check_calibration_spread.py` is a library file with no `__main__` block — running it directly does nothing. Use `run_calibration_diagnostic.py`.

---

## What happens, in order

1. **AMP classifier** — trained on `amp_pos_clf` / `amp_neg_clf` (3:1), calibrated on validation, evaluated **once** on the sealed test split → `amp_metrics.json`
2. **CPP classifier** — same pattern → `cpp_metrics.json`
3. **Generator pool** — loaded separately from `amp_gen_path` / `cpp_gen_path`, combined and deduplicated; the CVAE trains on this pool, *not* the classifier split
4. **Score-guided generation** — 5 rounds of latent sampling, scoring, softmax-weighted seed selection (T = 0.3), 1–3 substitutions per seed
5. **Biological filters** — charge, GRAVY, instability, hydrophobic moment, aggregation propensity
6. **Diversity control** — greedy edit-distance filter (80 % identity) then K-means (k = 20), top-4 per cluster
7. **Evaluation** — novelty and five-population score comparison on the final set

### Biological filter thresholds

| Criterion | Threshold | Rationale |
|---|---|---|
| Length | 8 – 50 aa | Synthetic accessibility |
| Net charge @ pH 7.4 | ≥ +1.0 | Electrostatic attachment to anionic membranes |
| GRAVY | < 2.5 | Hydropathy ceiling |
| Instability index | < 60 | Solution stability |
| Hydrophobic moment | ≥ 0.0 | Helical amphipathicity |
| Aggregation propensity | < 5 | β-aggregation avoidance |
| `p_AMP`, `p_CPP` | ≥ 0.50 | Both activities above the calibrated threshold |

---

## Outputs

```
outputs/<timestamp>/
├── config_used.yaml          # Exact configuration for reproduction
├── candidates.csv            # Final ranked candidates (main output)
├── amp_metrics.json          # Sealed-test ROC/PR curves, Brier, calibration bins
├── cpp_metrics.json
├── score_comparison.csv      # Five-population matrix, calibrated scale
├── score_comparison_raw.csv  # Five-population matrix, raw scale
├── novelty_report.json       # Nearest-neighbour distance distribution
└── report.txt                # Human-readable summary
```

```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/TIMESTAMP/candidates.csv')
print(df[['rank','sequence','amp_score','cpp_score','combined_score']].head(10).to_string())
"
```

### `candidates.csv` columns

| Column | Description |
|---|---|
| `rank` | Rank by combined score — see the ranking-granularity caveat above |
| `sequence` | Amino-acid sequence (canonical alphabet) |
| `amp_score` / `cpp_score` | Calibrated ensemble probabilities |
| `combined_score` | Geometric mean √(amp × cpp) |
| `cluster` | K-means cluster ID (0–19) |
| `length`, `molecular_weight`, `pI`, `net_charge` | Basic properties |
| `instability_index` | Guruprasad |
| `GRAVY` | Kyte–Doolittle |
| `hydrophobic_moment` | Eisenberg (window 11, 100°/residue) |
| `aggregation_propensity` | Longest consecutive hydrophobic run |
| `aromaticity`, `aliphatic_index`, `boman_index`, `ww_hydrophobicity` | Additional descriptors |

---

## Reproducibility

Every run writes `config_used.yaml`. To reproduce exactly:

```bash
python main.py --config outputs/<timestamp>/config_used.yaml
```

Results reported here and in the manuscript are seed 42. Seeds 43 and 44 were run and produced independent dataset draws (AMP positive median length 17 / 18 / 18), but **metric variance across seeds has not been quantified** — a single-seed result should not be read as a stable estimate.

---

## Downstream validation — mandatory before any biological claim

The 80 candidates are **computational prioritisations only**. No sequence has been synthesised or assayed.

1. **MIC** — broth microdilution against *S. aureus* ATCC 29213 and *E. coli* ATCC 25922
2. **Haemolysis** — human erythrocyte lysis at 2× and 10× MIC
3. **Cell uptake** — confocal microscopy of FITC-labelled candidates in HeLa or RAW264.7
4. **Computational toxicity triage** — ToxinPred 3.0, HemoPI 2.0, AlgPred 2.0 as a minimal pre-synthesis screen

HemoPI2 is applied during *curation* as a generation-time filter, not as a training label: 39 % of AMP positives and 49 % of the 345 negatives are flagged, so filtering one class only would teach the classifier the HemoPI2 boundary, and filtering both would halve the negatives.

---

## Known limitations

**The CPP negative set is the most important open issue.** Of 217 training negatives, 42 carry direct experimental evidence of non-penetration and 175 are UniProt-derived, of which **90.2 % are annotated as secreted**. Because the tier-2 majority shares a subcellular provenance, the reported CPP separation of +0.382 cannot yet be attributed unambiguously to penetration rather than to secretory origin — the same class of confound removed on the AMP side. The diagnostic is cheap: score the 42 tier-1 and 175 tier-2 negatives *separately* under the trained ensemble. **This has not been done.**

**The old-versus-new comparison is confounded.** Four factors changed between the two configurations — positive set, negative set, class ratio, scoring pathway. The drop from ROC-AUC 0.996 to 0.785 is *consistent with* the negative-source substitution but not attributable to it. A controlled substitution (positives and ratio fixed, negative source varied across quantitative-inactive / annotation-absence / shuffled) is required before the curation strategy can be called demonstrated.

**Other limitations.**

- The AMP term is non-binding (see [above](#the-amp-result--read-this-before-using-the-candidates)).
- Absolute dataset sizes are small: 335 AMP negatives, 42 tier-1 CPP negatives.
- Isotonic calibration discretises the score scale; fine ranking is not supported.
- ESM-2 weights are frozen. Fine-tuning may improve representation but risks overfitting on a corpus this size.
- No structural prediction (AlphaFold2 / ESMFold). Worth adding for the longer candidates.
- Novelty statistics depend on reference-set size. `nearest_training_distance` defaults to `n_sample=200`, which **overstates novelty roughly threefold** — an earlier analysis reported 90 % novel against a 200-sequence subsample versus 38.75 % against the full 9,205 pool. Always pass the full pool and state the reference size.

---

## Findings worth reusing

Three incidental results from curation that affect anyone building AMP or CPP classifiers:

- **41 % aggregator over-inclusion.** 140 of 342 DBAASP-inactive peptides are simultaneously listed as antimicrobial by DRAMP 4.0, CAMPR4 or ADP v6. Only 29 are haemolytic and 111 sat in the non-haemolytic positive pool, so this is over-inclusion by the aggregators rather than genuinely lytic peptides among the inactives.
- **CPPsite 3 "negatives" are not non-CPPs.** Both of its classes are cell-penetrating peptides; the negatives are simply low-uptake (< 25 % of control). Using them as non-penetrating negatives trains a model on uptake efficiency, not on penetration.
- **`KW-0985` is Congenital erythrocytosis**, matching six proteins. It is not a cell-penetrating-peptide keyword — **no such UniProt keyword exists.** Any filter using it is a silent no-op. This repository substitutes sequence-based exclusion plus a keyword self-test that aborts on identifier/name mismatch.

---

## Outstanding

- [ ] Add the curation scripts (DBAASP download + inactivity parsing, UniProt mature-chain extraction, tier assembly, conflict resolution, length matching) — required for the manuscript's Data Availability statement to hold
- [ ] Mint the new Zenodo deposit for the recurated datasets and replace the DOI placeholders
- [ ] Split CPP negative scores by tier and report separately
- [ ] Controlled negative-source substitution with positives and class ratio held fixed
- [ ] Quantify metric variance across dataset seeds 42 / 43 / 44
- [ ] Commit `requirements.txt` / `environment.yml` if not already present

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'amp_gen_path'` | Old `config.yaml` predating the generator-path split | Update `config.yaml` to the current schema |
| `FileNotFoundError` on any of the six inputs | Config path does not match your layout | Fix the path or use `--set data.X=...` |
| CVAE trains on ~700 sequences instead of ~9,800 | Old `main.py` / `training/train.py` predating the generator-wiring fix | Update both files |
| No candidates pass the biological filters | Classifiers scoring low on generated sequences | `--set filters.min_amp_score=0.40 --set filters.min_cpp_score=0.40` for a first pass |
| Results directory looks empty | Confusing `output/` (inputs) with `outputs/` (results) | Check the plural |
| `python diag/check_calibration_spread.py` does nothing | Library file, no `__main__` block | Use `python run_calibration_diagnostic.py --config config.yaml` |

---

## How to cite

**Method article (in preparation):**

```bibtex
@article{upadhyay_2026_curation,
  author  = {Upadhyay, Aditya and [Co-authors]},
  title   = {Evidence-tiered negative-set curation and a cross-classifier
             specificity control for dual-activity peptide design},
  journal = {MethodsX},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

**Dataset:**

```bibtex
@dataset{upadhyay_2026_dataset,
  author    = {Upadhyay, Aditya},
  title     = {Curated AMP and CPP positive and negative sets, and
               generated candidate peptides},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {[to be minted]}
}
```

**Companion thesis chapter:**
Upadhyay A. *et al.* *Big data analytics using contemporary algorithms and data warehousing.* AcSIR, 2026.

---

## Affiliation

CSIR — Indian Institute of Chemical Biology, Kolkata, India
AcSIR — Academy of Scientific and Innovative Research

## License

MIT — see [LICENSE](LICENSE). The companion dataset will be released under CC-BY 4.0.
