# Reproduction and results package — 9 November 2026

Read the [root results summary](../../README.md) first. This snapshot contains the completed compute queue, including three documented numerical VaDER failures. It does not claim that every requested fit succeeded.

## Files

- `RankWalk_Evaluation_Report.pdf`: final 20-page LaTeX-compiled report.
- `RankWalk_LaTeX_Report.zip`: standalone LaTeX source, all vector PDF/PNG plots, score tables, environment metadata and failure logs.
- `report/`: the unpacked editable LaTeX package. Run `make -C report` to compile; no network access or ML environment is required.
- `report/data/`: consolidated simulation, primary-clinical, extension, K=4 and PAQUID-onset sensitivity scores.
- `per_run_scores/`: individual scalar CSV/JSON checkpoints. Pilot and excluded-input diagnostic files are also retained; filenames and scope fields identify them. Do not pool them into primary summaries.
- `benchmark_inputs.zip`: exact generated data and processed inputs. Contains 300 simulated CSVs (six settings × 50), 150 VaDER NPZ inputs and 12 derived long/survival tables from public clinical datasets. Archive members are under `data/`.
- `scripts/`: the actual Python/R analysis scripts, including simulation generators, model fits, clinical wrappers, metrics, controls, plots and report creation. These preserve historical cloud paths for provenance.
- `reference_sources/`: original hard simulation generator and the TAPIO R package sources required by Simulation 1 and TAPIO baselines, with their original attribution and license metadata. The rest of the repository remains under its original license; the vendored TAPIO sources retain their own licensing.

Large intermediate embeddings, RDS checkpoints, installed environments and third-party binary dependencies are not committed. Source dependency commits, environment versions and the TimesURL CPU portability patch are in `report/provenance/`. The original uploaded paper is referenced by checksum rather than redistributed.

## Build the report

```sh
cd evaluation/2026-11-09/report
make
```

A standard TeX Live installation is sufficient. `report.tex` uses relative paths, no shell escape and no online resources. The package was unpacked into a different directory and successfully rebuilt with two pdflatex passes.

## Prepare scientific reruns

The archived scripts used `/workspace/...` paths. Do not blindly run their historical queue scripts: they also contain historical process IDs. `prepare_local_run.py` creates relocated copies of the scientific scripts and unpacks the exact inputs into an ignored work directory. It does **not** install dependencies or start model fitting.

Install RankWalk from this checkout with the optional fast dependencies in a Python environment. R must contain the packages in `report/provenance/R-session.txt`, including native `fdapace`, `fda`, `kml3d`, `dtw`, `proxy`, `TAPIO`, `survival` and `Hmisc`. The old VaDER implementation requires a separate Python 3.9 environment with the exact dependencies in `vader-freeze.txt` (notably NumPy 1.19.5 / TensorFlow 2.7.1). The main Python environment is documented in `python-freeze.txt`. Package installation on a new platform may require different build tools; exact trained clinical graph refits retain the documented NumPy graph-RNG limitation.

```sh
python evaluation/2026-11-09/prepare_local_run.py \
  --python /absolute/path/to/scientific-python/bin/python \
  --vader-python /absolute/path/to/vader-python39/bin/python \
  --rscript /absolute/path/to/Rscript \
  --fetch-dependencies
```

This clones the pinned public TS2Vec, TimesURL, VaDER, JM, joineR and joineRML reference sources into `work/references/` and applies the recorded TimesURL CPU-only allocation patch. Native TAPIO R sources are copied into `work/TAPIO`; install this R package into the chosen R library if it is not already installed. `local_run.json` records the resolved paths. Preparation without `--fetch-dependencies` is useful for offline inspection; cloning does not install Python/R packages.

Do not execute the entire queue without planning compute: the evaluation took hours on four shared CPU cores. The saved scalar CSVs are sufficient for inspecting the final results. For fresh fits, run the **relocated copies under `work/scripts/`**, with the chosen interpreters and these environment settings:

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
```

Suggested execution order:

1. Inputs are already unpacked. `Rscript work/scripts/generate.R` checks existing files and skips them; use a separate empty data directory if deliberately regenerating from the original R generators.
2. `python work/scripts/run_rankwalk.py` runs all six settings with 50 repetitions each; scenario names can also be passed as arguments. `Rscript work/scripts/run_baselines.R` runs native R comparators. `python work/scripts/run_encoders.py` runs TS2Vec and exports VaDER inputs. `python work/scripts/vader_chain.py` dispatches VaDER using the configured Python 3.9 interpreter.
3. `Rscript work/scripts/clinical_PBC2.R`, `clinical_AIDS.R`, `clinical_HEART.R` and `clinical_PAQUID.R` reproduce native clinical settings. They checkpoint to `work/results/` and skip complete rows on resume.
4. `Rscript work/scripts/export_clinical.R new` prepares the additional public cohorts; calling it without `new` exports original-cohort inputs from native clinical checkpoints. The exact exported tables from the completed evaluation are already included in the input archive, so this step is optional when reusing those inputs.
5. `python work/scripts/real_chain.py` runs ten extension fits per cohort/method. `run_controls.py` adds visit-time/count and marker-summary controls. Individual extension fits can be run with `real_one.py COHORT METHOD REP`, using the VaDER interpreter for VaDER.
6. `Rscript work/scripts/clinical_sensitivity.R` computes cluster-count and endpoint checks on newly saved native representations. `repository_diagnostics.R` and `ts2vec_latent_diagnostic.py` reproduce the explicitly labeled source-input leakage diagnostic. These are not valid observed-only comparator scores.

The source reporter and historical queue scripts remain available for audit; they are not part of the relocated execution entry points. The report's score tables already distinguish actual successful counts. VaDER native and posterior-readout summaries reuse the same model and must not be counted as independent fits. Model and environment artifacts are intentionally excluded from Git, and are generated locally during reruns.

## Endpoint and evaluation interpretation

The primary native PAQUID code has a time-zero endpoint defect. The extension uses `agedem` and mean pooling; the saved-representation endpoint correction is a third, separately labeled analysis. Source clinical settings and common extension settings differ, especially for HEART and PAQUID. Full-follow-up C-indices are descriptive association statistics, not future prediction or AUROC. Clinical SDs measure stochastic fit variation on the same patients, not patient-sampling uncertainty.

The historical provenance audit records the original instruction to push nothing and an all-successful flag of `false` due to VaDER failures. The user later authorized publication on branch `results` only; the audit is preserved as the original experiment snapshot. No library implementation changes or updates to `main` are part of this publication.
