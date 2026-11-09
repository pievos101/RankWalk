# RankWalk — completed evaluation results

This `results` branch contains the 9 November 2026 paper reproduction and expanded longitudinal benchmark. It starts from RankWalk commit `58a87af4fdcdd243425106622e0e4f46dcc1ba6a`; library implementation files are unchanged. The `main` branch is unchanged by this results publication. For installation, API details and the fast backend, see [README_IMPLEMENTATION.md](README_IMPLEMENTATION.md).

## Downloads and report

- [Compiled PDF report](evaluation/2026-11-09/RankWalk_Evaluation_Report.pdf)
- [Download the full LaTeX package, including every plot and data table](https://github.com/pievos101/RankWalk/raw/refs/heads/results/evaluation/2026-11-09/RankWalk_LaTeX_Report.zip)
- [Browse LaTeX source](evaluation/2026-11-09/report/report.tex), [plots](evaluation/2026-11-09/report/figures), and [consolidated CSV tables](evaluation/2026-11-09/report/data)
- [Actual simulation and evaluation scripts](evaluation/2026-11-09/scripts)
- [Per-run scalar results](evaluation/2026-11-09/per_run_scores)
- [Generated benchmark inputs](evaluation/2026-11-09/benchmark_inputs.zip) and [reproduction instructions](evaluation/2026-11-09/README.md)

The LaTeX ZIP compiles independently with `make`; the PDF is 20 pages. The benchmark input archive contains all 300 generated simulation CSVs, 150 preprocessed VaDER input files and 12 derived public clinical tables. Environment installations, model weights and large intermediate embedding files are excluded.

## Main findings

- **Clean Simulation 1:** RankWalk has the highest mean ARI among the tested methods, with a small advantage over FPCA (+0.011 ARI; exploratory paired bootstrap 95% interval +0.003 to +0.019).
- **Corrupted-marker setting:** RankWalk and weighted TAPIO are effectively tied. RankWalk is much more robust than FPCA in this setting.
- **Regime-switching Simulation 2:** observed-marker TS2Vec outperforms RankWalk (0.961 versus 0.911 ARI). RankWalk exceeds the other tested methods.
- **PBC2:** RankWalk leads the expanded comparison, including the observation-process and simple-summary controls.
- **PROTHRO and RENAL:** RankWalk leads the tested biomarker representations, but visit-time/count controls score higher. Full-follow-up concordance therefore cannot establish prospective prediction or biomarker-specific benefit.
- **PAQUID:** corrected-endpoint results favor simple marker summaries and TimesURL over the common mean-pooling RankWalk configuration.
- **Configuration matters:** the paper-specific HEART configuration reproduces C-index 0.706, while the common extension configuration gives about 0.581. Moving Simulation 1 to a subject-shared random curve also changes its difficulty substantially.

These results support competitive, setting-dependent performance; they do not demonstrate universal superiority or an exhaustive 2026 state-of-the-art frontier.

## Primary simulation comparison

Mean **ARI ± sample SD**. Each cell uses 50 generated datasets, except VaDER Simulation 2: 49 successes and one numerical failure. Known labels are used for evaluation only.

| Method | S1 clean | S1 corrupted marker | S2 regime switching |
|---|---|---|---|
| RankWalk | 0.924 ± 0.039 | 0.936 ± 0.100 | 0.911 ± 0.080 |
| FPCA | 0.913 ± 0.038 | 0.644 ± 0.294 | 0.654 ± 0.021 |
| DTW | 0.888 ± 0.047 | 0.681 ± 0.246 | 0.839 ± 0.068 |
| kml3d | 0.888 ± 0.043 | 0.869 ± 0.136 | 0.659 ± 0.021 |
| TAPIO PC1 | 0.666 ± 0.094 | 0.820 ± 0.169 | 0.676 ± 0.032 |
| TAPIO weighted | 0.811 ± 0.082 | 0.938 ± 0.092 | 0.681 ± 0.033 |
| TS2Vec (observed markers) | 0.573 ± 0.177 | 0.642 ± 0.227 | 0.961 ± 0.053 |
| TS2Vec (source serialization) | 0.770 ± 0.077 | 0.684 ± 0.213 | Excluded: latent-state source input |
| VaDER (posterior + k-means) | 0.520 ± 0.156 | 0.563 ± 0.215 | 0.660 ± 0.035 |


TS2Vec source serialization is a separate sensitivity, not another independent architecture. Simulation 2 source-input leakage diagnostics are excluded from this valid observed-marker comparison. Clean and noisy Simulation 1 use different base-noise settings, so a higher noisy-setting score does not imply that adding noise improves clustering.

## All 300 RankWalk fits

| Setting | Fits | ARI ± SD |
|---|---|---|
| S1 regular, clean | 50 | 0.924 ± 0.039 |
| S1 corrupted marker | 50 | 0.936 ± 0.100 |
| S1 irregular | 50 | 0.905 ± 0.043 |
| S2 regular | 50 | 0.911 ± 0.080 |
| S2 irregular | 50 | 0.921 ± 0.090 |
| S1 subject-shared curve sensitivity | 50 | 0.782 ± 0.081 |


The subject-shared curve setting follows the paper description rather than the repository per-visit redraw. FPCA in this sensitivity gives 0.756 ± 0.070. Irregular inputs use a documented common-grid interpolation correction; graph and readout settings also differ, so their effect is not solely sampling irregularity.

## Completed original clinical reproduction

Repository-specific settings and readouts are preserved. Mean **C-index ± SD** and **log-rank χ² ± SD** across fits on the same patients.

| Cohort | Method | Fits | C-index | Log-rank χ² |
|---|---|---|---|---|
| PBC2 | RankWalk | 20 | 0.788 ± 0.018 | 240.788 ± 26.954 |
| PBC2 | FPCA | 20 | 0.760 ± 0.000 | 127.498 ± 0.000 |
| AIDS | RankWalk | 20 | 0.689 ± 0.006 | 104.113 ± 9.392 |
| AIDS | FPCA | 20 | 0.642 ± 0.000 | 53.091 ± 0.000 |
| HEART | RankWalk | 20 | 0.706 ± 0.022 | 54.161 ± 8.408 |
| HEART | FPCA | 20 | 0.569 ± 0.000 | 4.806 ± 0.000 |
| PAQUID | RankWalk | 30 | 0.706 ± 0.015 | 40.197 ± 5.054 |
| PAQUID | FPCA | 30 | 0.659 ± 0.000 | 34.286 ± 0.000 |


PBC2/AIDS/HEART FPCA values match the paper after rounding. PAQUID source concordance also rounds similarly, but its FPCA χ² is 34.29 rather than the paper’s 5.06; this discrepancy is not resolved. **The native PAQUID endpoint assigns dementia events time zero and is preserved only as a source reproduction.**

## Expanded real-world comparison

Mean **C-index ± sample SD**, source K=3 except HEART K=2. All successful fits are summarized: 10 per method/cohort, except VaDER on RENAL and HEART (9). These columns are descriptive successful-fit means, not matched-subset significance tests. PAQUID uses the corrected `agedem` endpoint and the common mean-pooling configuration.

| Method | PBC2 | PAQUID | PROTHRO |
|---|---|---|---|
| RankWalk | 0.788 ± 0.027 | 0.682 ± 0.040 | 0.707 ± 0.010 |
| FPCA | 0.760 ± 0.000 | 0.676 ± 0.000 | 0.636 ± 0.000 |
| TS2Vec | 0.656 ± 0.057 | 0.613 ± 0.032 | 0.668 ± 0.034 |
| TimesURL | 0.716 ± 0.014 | 0.746 ± 0.022 | 0.679 ± 0.011 |
| VaDER (native mixture) | 0.681 ± 0.055 | 0.638 ± 0.089 | 0.632 ± 0.048 |
| VaDER (posterior + k-means) | 0.711 ± 0.014 | 0.706 ± 0.046 | 0.657 ± 0.024 |
| Simple marker summaries | 0.729 ± 0.001 | 0.763 ± 0.000 | 0.637 ± 0.000 |
| Visit-time/count control | 0.745 ± 0.000 | 0.682 ± 0.000 | 0.804 ± 0.000 |


| Method | RENAL | AIDS | HEART |
|---|---|---|---|
| RankWalk | 0.841 ± 0.011 | 0.689 ± 0.009 | 0.581 ± 0.054 |
| FPCA | 0.624 ± 0.000 | 0.642 ± 0.000 | 0.569 ± 0.000 |
| TS2Vec | 0.713 ± 0.046 | 0.526 ± 0.037 | 0.516 ± 0.063 |
| TimesURL | 0.741 ± 0.017 | 0.636 ± 0.005 | 0.521 ± 0.065 |
| VaDER (native mixture) | 0.719 ± 0.055 (n=9) | 0.617 ± 0.018 | 0.533 ± 0.030 (n=9) |
| VaDER (posterior + k-means) | 0.758 ± 0.039 (n=9) | 0.619 ± 0.018 | 0.535 ± 0.060 (n=9) |
| Simple marker summaries | 0.781 ± 0.000 | 0.641 ± 0.000 | 0.424 ± 0.000 |
| Visit-time/count control | 0.887 ± 0.000 | 0.835 ± 0.000 | 0.705 ± 0.000 |


VaDER native mixture and posterior-plus-k-means are **two readouts of the same fitted models**, not independent fits. A displayed zero SD means stable partitions on the same cohort; it is not zero patient-sampling uncertainty.

## Interpretation, deviations and failure audit

1. Clinical trajectories include full observed follow-up; clusters are ordered using observed outcomes (or evaluated with a fitted categorical Cox model in native PAQUID). **These are descriptive associations, not held-out or prospective predictive accuracy.** Time-only controls are especially strong on PROTHRO, RENAL, AIDS and HEART. A separate fixed-landmark, held-out design is needed for prognostic claims.
2. Simulation 1 repository code redraws quadratic random effects per visit; the paper describes a subject-shared curve. Both are analyzed separately.
3. Simulation 2 source preprocessing includes the unobserved simulator regime in competitor inputs. The main comparison removes it; source-input diagnostics are separately labeled in the report and CSVs.
4. Source clinical cluster counts are 3 (PBC2/AIDS/PAQUID) or 2 (HEART), whereas the paper text describes 4. K=4 sensitivity results are included.
5. Source PAQUID treats an eventual subject status as visit-specific dementia status, assigning event time zero. The common extension corrects the endpoint using `agedem`; it also changes the readout from padded flattening to mean pooling. These are distinct analyses.
6. Native clinical graph subspace randomness was not pre-seeded in NumPy; R clustering, Torch and the walk sampler were seeded. Saved outputs permit exact score replay; graph refits retain the documented reproducibility limitation.
7. Concurrent wall times on four shared CPU cores and unequal training budgets are not isolated comparative speed evidence. The fast backend was active; no new controlled old-versus-fast speedup factor is claimed.
8. Three VaDER fits failed despite same-seed retries: S2 regular replicate 9 (singular covariance), RENAL extension replicate 8 (mixture initialization covariance), HEART extension replicate 7 (singular covariance). Failed fits have no invented score or replacement seed. The compute queue finished, but the historical strict all-successful flag remains false. See the [failure audit](evaluation/2026-11-09/report/provenance/failed_runs.json).

Historical provenance records mention the original no-push instruction and original cloud paths; they were captured before the user subsequently authorized publishing this **results branch only**.

## Reproduction and files

See [evaluation/2026-11-09/README.md](evaluation/2026-11-09/README.md) for the layout, dependency versions, script relocation and execution order. The generated inputs and scalar per-run results are committed so the numeric summaries can be inspected without rerunning several hours of training. The LaTeX report can be rebuilt without installing any machine-learning environments.

The original implementation README remains available as [README_IMPLEMENTATION.md](README_IMPLEMENTATION.md).
