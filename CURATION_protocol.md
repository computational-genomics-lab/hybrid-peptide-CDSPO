# Curation protocol

Complete specification of how the AMP and CPP training sets in this deposit were
built. Written so the procedure can be reproduced or applied to a different
bioactivity task. Every non-obvious decision is recorded with its reason, because
several of them look arbitrary and are not.

**Code:** https://github.com/computational-genomics-lab/hybrid-peptide-CDSPO
**Data:** https://doi.org/10.5281/zenodo.21704606

---

## 0. The problem this protocol solves

Supervised models for peptide bioactivity need a negative class. Repositories of
experimentally confirmed *inactive* peptides are far smaller than repositories of
active ones, so the common expedients are to shuffle positives or to sample
proteins carrying no positive annotation. Both define the negative class by what it
lacks.

Any systematic difference between the sampling frames of the two classes — length,
amino-acid composition, subcellular provenance — is then available to the model as
a shortcut, and reported performance measures frame separability rather than
biology.

**Worked example from this project.** An early configuration trained the AMP
classifier against 43 UniProt cytoplasmic proteins selected by absence of an
antimicrobial keyword. It reported sealed-test ROC-AUC 0.996 and PR-AUC 1.000 — on
a test partition holding roughly six negatives. Random peptides scored 0.862 under
that model. What it had learned was *short cationic peptide* versus *long
cytoplasmic protein*. Replacing that negative class with 335 peptides carrying
quantitative inactivity evidence reduced ROC-AUC to 0.785, because the task became
better posed.

---

## 1. AMP positives

### 1.1 v1 — aggregate sources (used for published results)

Pooled from four repositories:

| source | contribution |
|---|---|
| DRAMP 4.0 | antibacterial, experimentally validated |
| DBAASP v3 | antibacterial, Gram-positive and Gram-negative targets |
| ADP v6 | antibacterial |
| CAMPR4 | antibacterial, experimentally validated |

Filter chain:

1. Pool and deduplicate by exact sequence → ~35,000
2. Remove sequences containing any residue outside `ACDEFGHIKLMNPQRSTVWY`
3. Length 5–50 → 15,283
4. Remove peptides predicted haemolytic by HemoPI2 → 9,317
5. Conflict resolution against tier-1 negatives (§4.1) → 9,205

**Length rationale.** Peptides below ~5 residues carry too little structural
information for sequence models; above ~50 the structural complexity increases
without a corresponding gain in prediction accuracy. Reported peak antimicrobial
activity lies around 10–30 residues, well inside the retained window.

Resulting distribution: mean length 17.2 ± 8.2, median 16, IQR 11–20.

**Design note — the disulfide restriction was abandoned.** An earlier version
restricted positives to disulfide-bonded ADP v6 entries. This biased the positive
class toward cysteine-rich defensin-like scaffolds and propagated that bias into
generated candidates, which showed a spurious bimodal length distribution with a
second mode at 38–45 residues. Removing the restriction removed the mode.

### 1.2 v2 — MIC-verified classifier positives (refinement, not yet published)

**Scope: this changes the AMP classifier positives only.** The generator pool stays
at 9,205 (§1.1), the negatives stay at 335 (§2), and all CPP sets are unchanged.

v1 classifier positives carry no potency information — a peptide with MIC 4 µg mL⁻¹
and one with MIC 200 µg mL⁻¹ receive the same label — and are a seed-dependent
length-matched sample rather than a stated rule. v2 fixes both by drawing positives
from the **same DBAASP parser used for the negatives** (§2) with the comparison
inverted.

1. Take `verdict == reject_active` rows from `dbaasp_audit.csv` — peptides rejected
   as negatives because their minimum converted MIC fell below 256 µg mL⁻¹. All have
   already passed the ≥ 3 species requirement. → 9,631
2. Drop 15 rows with `min_MIC_ugml == 0` (physically impossible; a unit-parse
   failure or a literal zero in the source) → 9,616
3. Length 5–50, canonical residues only, deduplicate → 8,469
   (removes modified peptides that had MW assigned by RDKit from SMILES)
4. **`min_MIC_ugml ≤ 4` and `n_species ≥ 8`** → **964**

Resulting set: MIC median 1.74 µg mL⁻¹ (IQR 0.81–2.67, max 4.00), species coverage
median 10 (range 8–48), length mean 20.0, median 18.

**Threshold rationale.** Two constraints fix the cut.

*Separation.* Broth microdilution reproduces to ±1 two-fold dilution step. A
positive cut at 128 against a negative floor of 256 leaves the classes adjacent
within assay noise. A cut at 4 leaves six dilution steps of empty space.

*Class ratio.* The negatives are capped by evidence availability at 335 — DBAASP
does not contain more peptides meeting the inactivity rule. At the earlier ≤ 32 /
≥ 3 species cut the positives number 7,126, giving 21:1, and this project's own
runs found the AMP classifier ceases to discriminate at ~27:1. MIC ≤ 4 with ≥ 8
species yields 964, or **2.9 : 1** — the same regime as v1 but reached by a rule
rather than a random draw.

**Two known limitations.**

`min_MIC_ugml` is the minimum across species, so the rule reads *"potent against at
least one organism, tested against ≥ 8"* rather than *"potent against ≥ 8"*. True
broad-spectrum selection requires per-species MIC columns and a re-run against the
raw DBAASP JSON records.

The v2 positives are **not length-matched** to the negatives: Kolmogorov–Smirnov
D = 0.101, p = 0.011, with positives running longer in the tail. Medians are close
(18 against 17) and the effect is small, but length remains weakly available as a
shortcut. v1 was importance-weighted (§5.5) and does not have this. Applying §5.5 to
the 964 would remove it at the cost of sequences and of reintroducing a seed.

---

## 2. AMP negatives — quantitative inactivity

Source: the complete DBAASP corpus, ~25,069 peptides, retrieved through the REST
interface as one JSON record per peptide (resumable, rate-limited, with retry and
backoff; invalid JSON rejected before write).

### 2.1 Admission rule

1. **Parse at monomer level.** Multimeric entries store measurements in nested
   `monomers[]` records with an empty top-level sequence field. Verified against a
   known multimer (Distinctin).
2. **MIC measurements only.** MBC and IC₅₀ measure different endpoints and are
   discarded. Microbial targets only; bacteria by default, fungi optional.
3. **Normalise units** across the several µg/mL spellings present in the source,
   including the distinct Unicode MICRO SIGN (U+00B5) and GREEK SMALL LETTER MU
   (U+03BC), and µM variants.
4. **Convert to µg mL⁻¹.** Ranges reduce to their **lower bound** — the strictest
   test of inactivity. Inequality prefixes are stripped.
   `MIC[µg/mL] = MIC[µM] × MW × 1e-3`
5. **Molecular weight policy**, in order:
   - canonical sequence → sequence MW (average masses)
   - modified peptide with valid SMILES → RDKit `Descriptors.MolWt` (**not**
     `ExactMolWt`)
   - neither available → **drop the measurement**, do not impute
   Sequence characters take precedence over the `unusualAminoAcids` flag.
6. **Convert, then count.** Unconvertible measurements are dropped *before*
   species counting, so coverage is never inflated by measurements that could not
   be used.
7. **Pool by sequence** across accession identifiers; provenance recorded in the
   audit as `pooled_from_multiple_ids`.
8. **Species requirement:** ≥ 3 distinct species. Strain designations collapse to
   genus + species (three *E. coli* strains count once). Abbreviated genera
   ("E. coli") are dropped as ambiguous.
9. **Inactivity threshold:** minimum converted MIC ≥ 256 µg mL⁻¹.
10. **Startup self-test.** Melittin at 1 µM must convert to 2.85 µg mL⁻¹. The run
    aborts if the formula is broken. Verified to catch an inverted formula.

Buffer = 0 % is canonical. Sensitivity buffers of ±10 % and ±15 % are reported as
separate runs rather than baked into the definition.

### 2.2 Yield

| stage | n |
|---|---|
| ≥ 3 species (canonical rule) | 345 |
| ≥ 2 species | 571 |
| ≥ 1 species | 671 |
| after length 5–50 | 342 |
| after canonical-residue filter | **335** |

The ≥ 2 and ≥ 1 counts are reported as a **pre-committed sensitivity analysis**,
not used to enlarge the set.

### 2.3 Audit output

`dbaasp_audit.csv`, 19,765 rows, one per peptide, columns: `sequence`, `length`,
`n_species`, `min_MIC_ugml`, `n_measurements`, `mw`, `mw_source`, `dbaasp_ids`,
`pooled_from_multiple_ids`, `dropped_measurements`, `abbrev_genus_dropped`,
`fungal_measurements`, `nonmicrobial_measurements`, `species_list`, `verdict`.

| verdict | n | meaning |
|---|---|---|
| `reject_active` | 9,631 | min MIC < 256 — these become v2 positives |
| `reject_few_species` | 8,969 | < 3 species after collapsing |
| `reject_length` | 823 | outside 5–50 |
| `keep_inactive` | 342 | admitted as negatives |

5,000 of the `reject_few_species` rows have `n_species = 0`, meaning every
measurement was dropped — largely the MW-unavailable cohort.

### 2.4 Incidental finding — aggregator over-inclusion

140 of the 342 DBAASP-inactive peptides (41 %) are simultaneously listed as
antimicrobial by DRAMP 4.0, CAMPR4 or ADP v6. Only 29 are haemolytic, and 111 were
present in the non-haemolytic 9,317 positive pool. The discrepancy therefore
reflects over-inclusion by the aggregating databases, not mispublished lytic
peptides. Where a conflict occurred the quantitative MIC measurement was treated as
decisive and the sequence removed from the positive set.

---

## 3. CPP positives

Pooled from CPPsite 1, 2, 3 and the CellPPD benchmark distributions.

| stage | n |
|---|---|
| raw records | 1,596 |
| deduplicated | 856 |
| non-haemolytic (HemoPI2) | 673 |
| length 5–30 | 660 |
| minus 8 sequences their source studies classify as non-penetrating | **652** |

**Rule applied throughout: an experimental designation supersedes an aggregated
database annotation.** All eight removed sequences originated in CPPsite 1.

A trailing column in the CellPPD distribution recording *predictions of the CellPPD
tool* rather than experimental labels was discarded before use.

---

## 4. CPP negatives — two evidence tiers

### 4.1 Tier 1 — experimentally tested non-penetrating

| study | sequences |
|---|---|
| Sanders 2011b | 34 |
| Dobchev 2010 | 24 |
| Hansen 2008 | 19 |
| Hällbrink 2005 | 16 |

Unique after deduplication: **43** (27 appear in two or more sets). After the 5–30
length window: **42**. The one loss is a 36-residue sequence.

These are predominantly truncated or point-mutated analogues of known CPPs shown
not to enter cells, and are therefore **hard negatives** — chemically adjacent to
the positive class.

### 4.2 Excluded sources, with reasons

| source | why excluded |
|---|---|
| CPPsite 1 negatives | randomly generated from Swiss-Prot, never tested |
| CPPsite 2 negatives | same |
| CPPsite 3 negatives | **low-uptake CPPs, not non-penetrating peptides.** Both classes are CPPs; positives are high-uptake (> 75 % of control), negatives low (< 25 %). Used here only as an exclusion filter |
| Sanders 2011c | 111 sequences resampled from the original 34; no independent information |

### 4.3 Tier 2 — UniProt mature chains

Query: reviewed, non-fragment entries; mature CHAIN and PEPTIDE features **excised
from longer precursors** rather than taking short entries directly, which avoids
the hormone/toxin/AMP enrichment characteristic of short Swiss-Prot records.

Excluded keywords: Antimicrobial, Antibiotic, Toxin, Membrane, Transmembrane.

**Keyword self-test.** The run aborts on any identifier-to-name mismatch. Added
after discovering that `KW-0985`, used in an earlier version as a
cell-penetrating-peptide exclusion, is in fact **Congenital erythrocytosis** and
matched six proteins — making the filter a silent no-op. **There is no UniProt
keyword for cell-penetrating peptides.** A related comment error was corrected at
the same time: `KW-0964` is *Secreted*, not *Transmembrane* (the query behaviour
was already correct).

Yield: 3,238 candidates → 1,897 unique at length 5–30 (median 18) → **1,251** after
redundancy reduction.

**Intra-set redundancy reduction at 80 % identity** removed 646 sequences, of which
643 were internal near-duplicates, 2 were ≥ 80 % identical to a positive, and 1 was
a known positive. This step is necessary rather than cosmetic: 186 of the 238
sequences of length 24 were orthologues of the aspartate 1-decarboxylase β-chain,
so a single protein family would otherwise have contributed roughly a tenth of the
pool.

Compartment distribution:

| stratum | before | removed | after | share |
|---|---|---|---|---|
| secreted | 1,650 | 522 | 1,128 | **90.2 %** |
| cytoplasm | 238 | 121 | 117 | 9.4 % |
| nucleus | 8 | 3 | 5 | 0.4 % |
| mitochondrion | 1 | 0 | 1 | 0.1 % |

A membrane stratum returned nothing — the query was self-contradictory
(`SL-0162 AND NOT KW-0472`) — and was dropped.

Note that redundancy reduction **concentrated** the secretion bias (87 % → 90.2 %).
This is disclosed rather than corrected; see §5.4.

---

## 5. Assembly decisions

Each of these looks arbitrary and is not. Do not silently revert them.

### 5.1 Generator and classifier train on different data

- **Classifier:** negatives capped at 3× positives.
- **Generator:** full positive pool, no negatives (a CVAE never sees them).

At the natural pool ratio of ~27:1 the AMP classifier ceases to discriminate. 3:1
is the regime in which the CPP classifier operates. A contamination guard asserts
that no generator positive appears in the tier-1 negatives.

### 5.2 Identity filtering is tier-gated

- **Tier 1: never filtered against positives.** A tested-inactive near-analogue of
  an active peptide is the single most informative negative available. Removing it
  for similarity discards exactly the sequences that define the boundary.
- **Tier 2: filtered at 80 %.** The label there is an inference, so a near-duplicate
  of a positive is more likely to be a mislabelled positive than a hard negative.

### 5.3 Conflict resolution is tier-dependent

- **Tier 1 conflict → negative wins**, positive removed. A quantitative MIC
  measurement supersedes an aggregator listing. 140 AMP positives removed
  (15,282 → 15,142; generator pool 9,316 → 9,205 after removing 111 tier-1
  inactives).
- **Tier 2 conflict → positive wins**, negative removed. The tier-2 label is an
  inference and yields to a curated positive.

### 5.4 No compartment balancing

Post-redundancy cytoplasmic sequences have median length 24 against the positives'
median of 14. Balancing on provenance would trade a secretion shortcut for a length
shortcut. The chosen alternative: **length-match across the whole pool and disclose
the 90.2 % secreted fraction**.

### 5.5 Length matching by importance weighting

Sampling weight `w = p_target(L) / p_pool(L)`, **not** `p_target(L)` alone — the
latter leaves the sampled mass dependent on pool composition. Smoothing constant
`0.05 / n_distinct_lengths`. Verified: negative median length tracks positive
median exactly (17/17 for AMP, 14/13 for CPP).

### 5.6 HemoPI2 is a generation-time filter, not a training label

Prevalence: 39 % of AMP positives, 49 % of the 345 negatives.

- Filtering **one class only** lets the classifier learn HemoPI2's decision
  boundary instead of antimicrobial activity.
- Filtering **both** would halve the negatives.

Therefore: **filter neither during classifier training.** Apply HemoPI2 to the
generator pool, and screen candidates at the end of the biological-filter stage —
after the p ≥ 0.5 thresholds, before edit-distance and K-means, because screening
after clustering breaks the 4-per-cluster × 20 structure.

---

## 6. Final dataset composition (seed 42)

```
AMP  conflicts resolved      140 positives removed  (15,282 → 15,142)
     generator pool          111 tier-1 inactives removed  (9,316 → 9,205)
     classifier              1,005 pos / 335 neg = 3:1,  length median 17/17
     experimental negatives  335/335 = 100 %

CPP  tier 1                  42
     tier 2                  1,897 → 1,251  (643 redundant, 2 identity, 1 overlap)
     classifier              651 pos / 217 neg = 3:1  (42 tier-1 + 175 tier-2)
     experimental negatives  42/217 = 19 %
```

Seeds 42, 43 and 44 were run. AMP positive median length 17 / 18 / 18 confirms
independent draws. Metric variance across seeds has not been quantified.

---

## 7. Code defects found and fixed

Recorded because each produced silent, plausible-looking output.

| defect | effect | fix |
|---|---|---|
| `KW-0985` used as CPP exclusion | matched *Congenital erythrocytosis*, 6 proteins — filter was a no-op | removed; sequence-based exclusion + keyword self-test |
| `KW-0964` comment read *Transmembrane* | comment only; query behaviour correct | comment corrected to *Secreted* |
| PepBDB extractor `seq.replace('X','')` | spliced flanking fragments into sequences that never existed | reject any sequence containing X |
| Headerless CSV read with `DictReader` | promoted row 1 to a column name, silently eating one sequence per file — including 1 of the 42 tier-1 non-CPPs | header hints widened; explicit headerless handling |
| `nearest_training_distance(n_sample=200)` | silently subsampled the reference set, overstating novelty ~3× (90 % vs 38.75 % against the full 9,205 pool) | pass full pool; always report reference-set size |
| `cv=3` isotonic calibration on training folds | optimistic probabilities via fold-internal leakage through shared feature scaling | `cv='prefit'` on a held-out partition |

---

## 8. Validating the result — cross-classifier specificity control

Curation alone does not prove a classifier learned the intended biology. Score five
populations under **both** trained classifiers:

1. generated candidates
2. training positives for the task
3. AMP negative set
4. CPP negative set
5. random baseline of uniform composition at the positives' length distribution

**Reading 1.** A classifier that learned its activity assigns low scores to its own
negatives and *not* to the other task's negatives. If both negative sets collapse
under both classifiers, the models share an axis of general peptide atypicality
rather than encoding distinct biology.

**Reading 2.** Comparing training positives against the random baseline tests
whether the classifier orders real actives above sequence noise. **A classifier can
pass reading 1 and fail reading 2**, so both must be reported.

Run on **both** the raw and the calibrated scale — isotonic calibration can compress
the dynamic range on one task and not another (here: AMP validation-set unique
values 201 → 73; CPP 131 → 42).

Result for this dataset: reading 1 passes (AMP model 0.366 on own negatives vs
0.744 on CPP negatives; CPP model 0.235 vs 0.653). Reading 2 fails on the AMP axis
— random peptides 0.763 vs training positives 0.753. The AMP term is therefore
non-binding during generation. This is the failure mode the control exists to
detect.

---

## 9. Reusing this protocol

The transferable parts, in order of generality:

1. **Build negatives from measured inactivity where a repository records it.** The
   admission rule in §2.1 transfers to any activity with a quantitative endpoint.
2. **Tier negatives by evidence strength and gate the downstream rules on the
   tier** (§5.2, §5.3). Tested negatives and inferred negatives should not be
   treated identically.
3. **Record every rejected sequence with its reason.** The audit files made the
   v2 positive set possible months after the fact, at no extra cost.
4. **Self-test unit conversions and keyword identifiers at startup**, and abort on
   failure. Both classes of defect in §7 were silent.
5. **Run the specificity control** (§8) before trusting any reported metric.

The protocol assumes a repository recording quantitative inactivity. DBAASP
provides this for antimicrobial peptides. Where no equivalent exists, the tier-2
approach — with its acknowledged provenance risk — is the only option available,
and the provenance composition should be disclosed as in §4.3.
