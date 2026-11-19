# Propagation Event Conditioned Multi Fidelity Neural Operator

This repository contains the implementation of CMF-PENO-RF for received-power field estimation and probabilistic 3D radio monitoring. The code covers propagation-event construction, multi-fidelity supervision, local and Fourier residual operators, variance estimation, coverage calculation and internal reference-based candidate-site planning.

## Setup

Use Python 3.11 or later. The shell runners target Bash on Linux or WSL and the full-model experiments use PyTorch. Install the dependencies and verify the basic experiment code:

    python -m pip install -e ".[full,test]" matplotlib
    python -m pytest -q tests/test_cmf_peno_data.py tests/test_propagation_events.py tests/test_calibration.py

For the independent ray-tracing experiment, use a separate environment with Sionna RT 2.2.0, Mitsuba 3.9.1 and Dr.Jit 1.5.0. See the solver script for its command-line options.

## Experiment entry points

| Paper component | Entry points in scripts/ |
| --- | --- |
| Local-scene inputs and event tensors | build_jiangxi_validation_scenes.py, build_jiangxi_grouped_folds.py, build_cmf_peno_rf_dataset.py |
| Nanchang evaluation and province grouped evaluation | run_cmf_peno_rf_v3_loso.sh, run_cmf_peno_rf_v3_province.sh |
| Network training, evaluation and variance calibration | train_cmf_peno_rf_v3.py, evaluate_cmf_peno_rf_v3.py, fit_cmf_peno_variance_temperature.py |
| Component comparisons and validation-selected configurations | run_joint_ablation_revision.py, evaluate_joint_ablation_revision.py, analyze_joint_ablation_revision.py, summarize_multimetric_ablation_revision.py |
| Candidate-site planning | evaluate_jiangxi_candidate_planning.py, summarize_jiangxi_planning.py, audit_planning_domain_revision.py |
| Reference runtime benchmark | run_fair_runtime_benchmark.sh, benchmark_reference_vs_network.py, summarize_reference_runtime.py |
| Independent reference and label adaptation | prepare_independent_rt_experiment.py, run_independent_sionna_rt.py, summarize_independent_rt.py, adapt_sparse_independent_rt.py |

Internal run keys retain the names used in the original experiment artifacts. In particular, `full_v3` identifies the complete CMF-PENO-RF configuration. The main grouped experiments use training seeds 2026, 2027 and 2028. The separate independent RT label adaptation uses seed 20261107.

For an existing province fold dataset tree, run from the repository root:

    FOLD_ROOT=/path/to/folds ROOT=/path/to/new_results DEVICE=cuda bash scripts/run_cmf_peno_rf_v3_province.sh

The fold directory must contain one `cmf_peno_rf_s32.npz` file per geographical group, under that group's directory. The Python entry points document input and output arguments through `--help`. Some analysis scripts default to the experiment archive's original paths. Provide the corresponding local paths and prior-stage outputs for those scripts.

## Data access

This release contains source code and experiment entry points. Access to the study datasets is governed by the manuscript's Data Availability statement. Running the paper experiments requires the corresponding scene inputs and, for evaluation of frozen models, trained checkpoints.
