# Project Handoff: Human Activity Recognition

## Current state

This is a working script-based scikit-learn experiment for classifying six activities from smartphone sensor data.

```text
6 classifiers × 3 feature-condition labels = 18 experiments
Best recorded run: Logistic Regression + RFE, 350 features, 96.23% test accuracy
```

## How to run it

From the repository root:

```bash
venv\\Scripts\\activate
python main.py
```

Execution order:

1. `src/data_loader.py` loads `dataset/train.csv` and `dataset/test.csv`, removes `subject` and `Activity` from the feature matrices, and extracts `Activity` as the target.
2. `run_all_experiments()` creates a fresh selector and model pipeline for every model/feature-condition combination.
3. Each pipeline is fitted on training data and predicts both training and test data.
4. Metrics, classification reports, selected-feature count, and timings are saved to `results/experiment_results.csv`.
5. `evaluation()` reads that CSV, prints the best/top runs and comparisons, and writes CSV summaries and PNG plots under `results/evaluation/`.

The loaders use repository-relative Windows-style paths. Run from the project root; portability to other operating systems may require replacing them with `pathlib.Path` handling.

## Source modules

- `src/data_loader.py` — train/test loading and X/y extraction. The public functions are currently named `feture`, `test_feture`, `target`, and `test_target`.
- `src/feature_selection.py` — `all_features_fs`, `f_classif_selection`, `mutual_info_selection`, and `rfe_feature_selection`.
- `src/models.py` — six pipeline factories: Logistic Regression, KNN, RBF SVM, Extra Trees, Random Forest, and LDA.
- `run_experiments.py` — model/selector registries, `N_FEATURES = 350`, training loop, and CSV persistence.
- `evaluation.py` — result loading, accuracy-gap calculation, summaries, and plots.
- `main.py` — end-to-end entry point.

## Important implementation note

Although the condition is named `all_features`, `run_experiments.py` passes `k=350` to every selector. The current condition therefore uses `SelectKBest(f_classif)` with 350 features, not all 561 raw features. If a true all-feature baseline is required, update the runner and documentation together so that this condition bypasses selection or uses `k=561`.

`mutual_info_selection()` exists but is not included in the current experiment matrix.

## Generated artifacts

```text
results/experiment_results.csv
results/evaluation/all_results.csv
results/evaluation/model_comparison.csv
results/evaluation/feature_selection_comparison.csv
results/evaluation/top_experiments.csv
results/evaluation/test_accuracy.png
results/evaluation/train_vs_test_accuracy.png
results/evaluation/training_time.png
results/notebook.ipynb
```

## Reproducibility and cautions

- Do not edit the source dataset files as part of the workflow.
- Test data is held out for evaluation; preprocessing and selection are fitted inside each training pipeline.
- Rerunning `python main.py` overwrites the experiment and evaluation outputs.
- Timing and results can vary with hardware and library versions.
- Dependencies are listed in `requirements.txt` (`pandas`, `scikit-learn`, and `matplotlib`).

## Recommended follow-up work

1. Rename the misspelled loader functions and update imports.
2. Separate the true all-feature baseline from the 350-feature F-Classif condition.
3. Add tests for data shapes, selector configuration, and result schema.
4. Add confusion-matrix reporting for class-level error analysis.
