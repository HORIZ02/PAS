# PAS
Project_PAS
# Spatial Habitat Radiomics for Placenta Accreta Spectrum

Analysis code supporting the manuscript:

> **Quantitative placental MRI features to discriminate placenta accreta spectrum from isolated
> placenta previa** — a retrospective, single-centre diagnostic accuracy study in 202 consecutive
> singleton pregnancies with placenta previa (101 placenta accreta spectrum, 101 isolated placenta
> previa).

The study asks whether quantitative features of placental **size, contour irregularity and
intraplacental signal heterogeneity** on a single sagittal T2-weighted MRI sequence discriminate
placenta accreta spectrum (PAS) from isolated placenta previa (IPP), and how much of that
discrimination survives removal of features correlated with placental volume.

**This repository contains code only.** It contains no patient data, no imaging and no patient
identifiers. See the manuscript's Data Availability statement.

---

## Repository layout

```
scripts/
├── 01_k_identifiability_probe.py            K identifiability (4 criteria x 2 feature spaces)
├── 20_habitat_segmentation.py               habitat segmentation (upstream)
├── 03_habitat_spatial_features.py           spatial feature extraction
├── 21_habitat_downstream_analysis.py        univariate / redundancy / incremental testing
├── 22_k_cross_task_comparison.py            K across tasks
├── 23_direction_determination.py            AUC direction handling
├── 40_result_summary.py                     traceable summary of every reported number
├── 41_definitive_analysis.py                OOF + bootstrap CI + DeLong + calibration/DCA
├── 42_number_traceability_check.py          number-by-number source check
├── 44_table_numbers.py                      Table 1 / Table 4 numbers
├── 57_roi_signal_qc.py                      ROI-vs-body signal QC (all 202 cases)
├── 62_contour_alignment_selfcheck.py        contour-vs-fill alignment self-check
├── 65_habitat_composition_k3.py             cohort-wide K=3 habitat composition
├── explore/                                 scans that informed the pre-specified families
│   ├── 10_imaging_features_increment.py
│   ├── 11_confounders_continuous_outcomes.py
│   └── 12_segmentation_sensitivity_volume_strata.py
└── figures/
    ├── _figlib.py                           shared colour / geometry library
    ├── figure1_study_flow.py                Fig. 1  study flow (STARD)
    ├── figure2_pipeline.py                  Fig. 2  method pipeline
    ├── figure3_habitat_partition.py         Fig. 3  habitat partition (K = 2/3/4)
    ├── figure4_performance.py               Fig. 4  ROC / calibration / decision curve
    ├── figure5_misclassification.py         Fig. 5  out-of-fold predictions, misclassified cases
    ├── figure6_auc_volume_adjusted.py       Fig. 6  AUC before vs after removing volume
    ├── figureS1_acquisition_heterogeneity.py    Fig. S1
    ├── figureS2_feature_correlation.py          Fig. S2
    ├── figureS3_clinical_forest.py              Fig. S3
    ├── figureS4_selection_stability.py          Fig. S4
    ├── figureS5_case_montage.py                 Fig. S5
    ├── figureS6_volume_tertiles.py              Fig. S6
    ├── figureS7_real_cases.py                   Fig. S7
    ├── figureS8_volume_signal_scatter.py        Fig. S8
    ├── figureS9_erosion_sensitivity.py          Fig. S9
    ├── figureS10_k_identifiability.py           Fig. S10
    ├── figure_graphical_abstract.py         graphical abstract
    └── figure_graphical_summary.py          graphical summary (sequential version)
```

`MANIFEST.md` lists the original script name of every file above, the scripts that were **not**
included (and why), and the privacy scan that was run before publication.

---

## Pipeline

The analysis runs in five ordered stages.

1. **K identifiability** (`01`) — for a case subset, evaluates whether the number of habitats *K* is
   identifiable, using four criteria (silhouette, Gaussian-mixture BIC, gap statistic with the 1-SE
   rule, ARI reproducibility) under two voxel feature spaces (intensity only; intensity + multi-scale
   local texture). A criterion whose optimum falls on a search-interval endpoint is treated as
   non-arbitrating.
2. **Habitat segmentation** (`20`) — mask cleaning → cropping → in-plane resampling to 1.0 mm →
   robust normalisation within the ROI → per-case k-means (K = 2/3/4) on the signal → habitat masks,
   plus a per-case QC table (habitat volumes, mean signal, k-means cut-points, consistency checks).
3. **Spatial feature extraction** (`03`) — per-habitat spatial and morphological descriptors:
   11 features per habitat plus inter-habitat descriptors (topology, fragmentation, spatial
   distribution), for a single modality (sagittal T2WI).
4. **Downstream analysis** (`21`, `22`, `23`) — slice-thickness dependence screen; redundancy
   spectrum against conventional geometry/shape features; univariate association for both endpoints
   (T1 = PAS vs IPP; T2 = poor prognosis); incremental testing with fold-internal top-*k* selection;
   matched-*k* comparison of the strict spatial pool against its morphological counterpart.
5. **Definitive analysis and reporting** (`41`, `40`, `42`, `44`) — case-level 5-fold
   cross-validation over 20 random seeds → out-of-fold predictions; bootstrap resampling of patients
   (1000 replicates) → 95% CIs; DeLong tests; calibration slope/intercept, Brier score and
   decision-curve analysis; then a traceable summary in which every reported number is annotated
   with the file it came from.

Quality control (`57`, `62`, `65`) and figure generation (`scripts/figures/`) are independent of one
another and can be re-run individually.

---

## Requirements

Verified under **Python 3.9.12**:

```
numpy        1.24.3
pandas       1.4.2
scipy        1.13.1
scikit-learn 1.0.2
nibabel      5.3.3
scikit-image 0.19.2
matplotlib   3.5.1
```

No deep-learning framework, no PyRadiomics and no SimpleITK are required: the descriptors are
computed directly with `scikit-image` / `scipy`, and the models are L2-penalised logistic
regressions from `scikit-learn`. DeLong's test is implemented in-house following Sun and Xu (2014).

---

## How to run

1. **Point the code at your working copy.** No personal path is hard-coded. Every script reads the
   project root from the environment variable `PAS_PROJECT_ROOT`:

   ```
   # Windows (cmd)          set PAS_PROJECT_ROOT=D:\my_project
   # Windows (PowerShell)   $env:PAS_PROJECT_ROOT = "D:\my_project"
   # macOS / Linux          export PAS_PROJECT_ROOT=/path/to/my_project
   ```

   The project root is expected to contain `使用的数据/` (images and masks), `结果_生境/`
   (result CSV files) and `features/`, mirroring the layout used for the manuscript.

2. **Inputs** (not distributed): de-identified sagittal T2-weighted NIfTI images, a placental ROI
   mask per case, and a clinical table with the eleven baseline variables used in Table 1.

3. **Run the stages in order:**

   ```
   python scripts/01_k_identifiability_probe.py
   python scripts/20_habitat_segmentation.py
   python scripts/03_habitat_spatial_features.py
   python scripts/21_habitat_downstream_analysis.py
   python scripts/41_definitive_analysis.py
   python scripts/40_result_summary.py
   ```

   Figure scripts read the CSVs written by the stages above and can be re-run individually, e.g.
   `python scripts/figures/figure4_performance.py` (`_figlib.py` sits beside them so `import _figlib`
   resolves).

---

## Methodological guarantees enforced in code

These are the points the manuscript turns on, and each is enforced by assertions rather than by
convention:

- **Fold-internal feature selection.** Features are ranked and selected *inside each training fold
  only*; selecting on the full cohort before cross-validation is treated as leakage.
- **Case-level splitting.** Cross-validation splits cases, never slices, so slices from one
  pregnancy never appear in both training and test sets.
- **Stated confidence-interval calibre.** Point estimates are the mean out-of-fold AUC over 20
  random seeds; the 95% CIs come from bootstrap resampling of patients, not from across-seed
  variation, and the calibre is stated wherever an AUC is reported.
- **Volume-independent pool.** A pre-specified control re-evaluates the feature pool restricted to
  features with |Spearman rho| < 0.5 against placental volume.
- **Segmentation-calibre control.** Signal features are recomputed after eroding the ROI by 5 and
  8 mm.
- **Matched-*k* comparison.** The strict spatial pool and its morphological counterpart are compared
  at equal numbers of selected features, with a guard that skips blocks too narrow to supply *k*.
- **Negative findings reported as such.** The number of habitats could not be identified and the
  spatial descriptors added no discrimination; both are reported as negative findings rather than
  being presented as an optimum.

---

## Data availability

The imaging and clinical datasets are **not** publicly available because they contain information
that could identify patients. De-identified data may be made available from the corresponding author
(Yan Su) upon reasonable request. This repository hosts the analysis code only.

---


## License

Released for academic and non-commercial research use. Please cite the manuscript above if you use
this code. If you intend to permit reuse, add an explicit license file (MIT, BSD-3-Clause or
CC-BY-NC-4.0) and update this section.
