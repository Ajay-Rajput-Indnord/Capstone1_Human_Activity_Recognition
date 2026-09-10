# Project Handoff: Human Activity Recognition

## Summary

This project recognizes six human activities from smartphone sensor data using Machine Learning.

The current experiment evaluates:

- 6 classifiers
- 3 feature-selection conditions
- 18 total classifier × feature-selection experiments

The three feature-selection conditions are:

- All Features
- F-Classif
- RFE

The F-Classif and RFE experiments select 350 features. The All Features condition uses all 561 features.

## Current project status

The completed experiment results are available in:

```text
results/experiment_results.csv
```

The reporting notebook is available in:

```text
results/notebook.ipynb
```

The six classifiers are:

1. Logistic Regression
2. KNN
3. RBF SVM
4. Extra Trees
5. Random Forest
6. LDA

The best recorded configuration is Logistic Regression with RFE and 350 selected features, with approximately 96.23% test accuracy.

## Current source modules

- `src/data_loader.py` loads train.csv and test.csv and separates sensor features from the Activity target.
- `src/feature_selection.py` provides All Features, F-Classif, and RFE helpers. It also contains a legacy Mutual Information helper that is not used in the current experiment conditions.
- `src/models.py` provides the six classifier pipeline factories.
- `src/evaluation.py` calculates accuracy and classification reports and writes result files.
- `main.py` is the root application entry point, but it is currently a legacy single-model entry point rather than the full experiment runner.


## Documentation changes completed

`architecture/design.md` now includes:

- The current six-classifier list.
- The current three feature-selection conditions.
- Function signatures, parameter types, and return values for every function in `data_loader.py`, `feature_selection.py`, `models.py`, and `evaluation.py`.
- The actual current project structure.
- A corrected Mermaid data-flow diagram.
- No references to nonexistent `run_experiment.py` or `src/main.py`.
- The current implementation limitations.

## Reproducibility

Run commands from the project root and use the local virtual environment:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

Run the follwing code:

```bash
python main.py
```
