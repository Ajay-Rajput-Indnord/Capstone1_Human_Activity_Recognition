# Design: Human Activity Recognition

## Overview

This project implements a Machine Learning system for Human Activity Recognition using smartphone sensor data.

The dataset contains 561 sensor features, a subject identifier and Activity target. The system predicts six activities:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

The current experiment compares 6 classifiers across 3 feature-selection conditions:

```text
6 classifiers × 3 feature-selection conditions = 18 experiments
```

The three feature-selection conditions are:

1. `All Features`
2. `F-Classif`
3. `RFE`

F-Classif and RFE are configured to use some features. All Features uses the complete 561-feature matrix.

## Current project structure

```text
Capstone_1/
├── architecture/
│   ├── design.md
│   └── proposal.md
├── dataset/
│   ├── train.csv
│   └── test.csv
├── results/
│   ├── experiment_results.csv
│   ├── notebook.ipynb
│   ├── graphs/
|   └── evaluation/
├── src/
│   ├── data_loader.py
│   ├── feature_selection.py
│   └── models.py
├── evaluation.py
├── main.py
├── run_experiments.py
├── requirements.txt
├── HANDOFF.md
└── README.md
```

## Architectural decisions

### Data source

The system uses the existing files:

- dataset/train.csv
- dataset/test.csv

The source CSV files must not be modified by preprocessing or training.

### Feature and target separation

The subject and Activity columns are removed from the model feature .

```text
X = all columns except subject and Activity
y = Activity
```

### Feature-selection conditions

#### All Features

Uses all 561 sensor features without reducing the feature matrix.

#### F-Classif

Uses `SelectKBest` with `f_classif` to rank features according to their relationship with the activity classes.

#### RFE

Uses Recursive Feature Elimination with a balanced Random Forest estimator. RFE repeatedly removes less important features until the configured number of features remains.

All selectors must be fitted using training data only. The fitted selector is then applied to test data.

### Classifier list

The six current classifiers are:

- Logistic Regression
- K-Nearest Neighbors
- RBF Support Vector Machine
- Extra Trees
- Random Forest
- Linear Discriminant Analysis

## Function signatures

The following signatures describe every function currently present in the four core source modules.

### `src/data_loader.py`

```python
def load_data() -> pandas.DataFrame
```

Loads and returns `dataset/train.csv` as a pandas DataFrame.

```python
def test_data() -> pandas.DataFrame
```

Loads and returns `dataset/test.csv` as a pandas DataFrame.

```python
def data_info() -> None
```

Loads the training data and prints its shape, information, and first five rows.

```python
def feture() -> pandas.DataFrame
```

Returns the training sensor features after removing `subject` and `Activity`.

```python
def target() -> pandas.Series
```

Returns the training `Activity` labels.

```python
def test_feture() -> pandas.DataFrame
```

Returns the test sensor features after removing `subject` and `Activity`.

```python
def test_target() -> pandas.Series
```

Returns the test Activity labels.

### `src/feature_selection.py`

```python
def all_features_fs() -> sklearn.feature_selection.SelectKBest
```

Returns a SelectKBest selector with `f_classif` and k="all", representing the All Features condition.

```python
def f_classif_selection(k: int = 350) -> sklearn.feature_selection.SelectKBest
```

Returns a SelectKBest F-Classif selector configured to retain k features.

```python
def rfe_feature_selection(k: int = 350) -> sklearn.feature_selection.RFE
```

Creates a balanced RandomForestClassifier estimator and returns an RFE selector configured to retain k features with step=10.

### `src/models.py`

For the model functions below, selector is a fitted feature selector and the return value .

```python
def extra_trees_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns transformation, the supplied selector, and a 500-tree `ExtraTreesClassifier`.

```python
def logistic_regression_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns the supplied selector and LogisticRegression using `C=1.0`, `solver="lbfgs"`, and `max_iter=5000`.

```python
def knn_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns a pipeline with StandardScaler the supplied selector, and distance-weighted KNN using seven neighbors.

```python
def rbf_svm_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns a pipeline with StandardScaler the supplied selector, and an RBF-kernel SVM using `C=10` and balanced class weights.

```python
def random_forest_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns a pipeline with StandardScaler, the supplied selector, and a 500-tree `RandomForestClassifier`.

```python
def lda_model(selector: sklearn.base.BaseEstimator) -> sklearn.pipeline.Pipeline
```

Returns a pipeline with StandardScaler the supplied selector, and `LinearDiscriminantAnalysis` using the SVD solver.

## Experiment matrix

Every experiment selects 350 features and records model name, feature-selection condition, train accuracy, test accuracy, training time, testing time, and classification reports in `results/experiment_results.csv`.

## Evaluation

The evaluation procedure uses:

- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Training time
- Testing time
- Number of selected features

Confusion-matrix visualization is a reporting improvement for the next stage.

## Data leakage prevention

The test dataset must not be used for:

- Fitting feature selectors
- Fitting preprocessing transformations
- Training classifiers
- Choosing hyperparameters

Selectors and preprocessing must be fitted on training data only and then applied to the test data.

## Results and reporting

The completed experiment results are stored in:

```text
results/experiment_results.csv
```

The reporting notebook is:

```text
results/notebook.ipynb
```

The best recorded configuration is Logistic Regression with RFE and 350 selected features, achieving approximately 96.23% test accuracy.

## Current implementation notes

1. `main.py` is a model entry point . 
2. The completed 18-experiment results were produced through the notebook workflow and are already persisted in `results/experiment_results.csv`.

## Reproducibility

- Keep the original CSV files unchanged.
- Run commands from the project root unless path handling is made independent of the working directory.
- Use the local virtual environment in venv/.
- Install dependencies from requirements.txt.

## Implementation sequence

1. Extract the notebook’s 18-experiment loop into a maintainable Python runner.
2. Add confusion-matrix calculation and reporting.
3. Keep result storage and notebook reporting separate from model implementation.
