# Human Activity Recognition

Machine Learning project for recognizing human activities using smartphone sensor data.

The project compares multiple Machine Learning classifiers and feature-selection techniques to determine which combination provides the best balance between predictive performance and computational efficiency.

## Project Objective

The dataset contains 561 sensor-derived features collected from smartphone accelerometer and gyroscope measurements.

The main objectives of the project are:

- Compare classifiers from different Machine Learning families.
- Compare the performance of using all features with feature-selection methods.
- Analyze the effect of feature selection on model performance and training time.
- Identify the best classifier-feature combination based on experimental results.


## Dataset

The project uses predefined training and testing datasets:

- `dataset/train.csv`
- `dataset/test.csv`

Each observation contains:

- **561 sensor-derived features**
- `subject` — participant identifier
- `Activity` — target variable

The `subject` column is kept separate from the sensor feature matrix as it identifies the participant rather than representing a sensor measurement.

The original dataset files are treated as read-only inputs and are not modified during the experiment.

## Activities

The dataset contains six human activities:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

## Feature Conditions

Each configured classifier is evaluated under three feature conditions.

### All Features

Uses all 561 sensor-derived features.

This acts as the baseline and allows us to measure whether feature selection improves performance or reduces computational cost.

### Anova

Uses Anova as the filter-based feature-selection method.

ANOVA identifies features that provide relevant information about the target activity classes.

It provides a fast and model-independent feature-selection condition.

### RFE

Uses Recursive Feature Elimination (RFE) as the wrapper-based feature-selection method.

RFE recursively removes less useful features using a configured estimator.

It provides a model-dependent feature-selection condition for comparison with Anova.

Feature selection is fitted using training data only and the fitted selector is then applied to the test data.

## Experiment Strategy

Every configured classifier is evaluated under all three feature conditions.

```text
Classifier 1
    ├── All Features
    ├── Anova
    └── RFE

Classifier 2
    ├── All Features
    ├── Anova
    └── RFE

...

Classifier N
    ├── All Features
    ├── Anova
    └── RFE
```

The total number of experiments is:

Number of classifiers * 3 Feature conditions

## Machine Learning Families

The project includes classifiers from:

- Linear-Based
- Tree-Based
- Kernel-Based
- Ensemble
- Discriminant

## Architecture

```text
Dataset
↓
Data Loading
↓
Data preprocess
↓
Feature / Target Separation
↓
Feature Selection
├── All Features
├── Anova
└── RFE
↓
Classifier
↓
Training
↓
Evaluation
↓
Result Storage
↓
Reporting Notebook
```

## Project Structure

```text
Capstone_1/
├── architecture/
│   ├── design.md
│   └── proposal.md
├── dataset/
│   ├── test.csv
│   └── train.csv
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   └── models.py
├── run_experiment.py
├── results/
│   ├── experiment_results.csv
│   └── notebook.ipynb
├── .gitignore
├── CHANGELOG.md
├── HANDOFF.md
├── README.md
└── requirements.txt
```
## Modules

### `data_loader.py`

Loads and preprocesses the training and testing datasets and prepares the feature and target data.

### `feature_selection.py`

Provides the Anova and RFE feature-selection methods.

### `models.py`

Creates and configures the Machine Learning classifiers.

### `evaluation.py`

Calculates accuracy, precision, recall, F1-score, confusion matrix, and training time.

### `run_experiment.py`

Runs the complete classifier × feature-condition experiment matrix and stores the results.

### `results/experiment_results.csv`

Stores the experiment results.

### `results/notebook.ipynb`

Used for reporting, visualization, comparison, and final analysis.

## Evaluation

Each experiment is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Time

The confusion matrix helps identify activities that are frequently confused.

Training time is recorded to compare computational cost.

## Data Leakage Prevention

The test dataset is used only for final evaluation.

It is not used for:

- Feature-selection fitting
- Model training
- Hyperparameter selection

Feature selectors are fitted using training data and then applied to both training and test features.

## Result Storage

Results are saved to:

```text
results/experiment_results.json
```
## Reporting

The reporting notebook reads the saved experiment results and is used for:

- Comparing classifiers
- Comparing feature conditions
- Comparing classifier families
- Visualizing accuracy
- Visualizing training time
- Displaying confusion matrices
- Analyzing per-class performance
- Identifying the best-performing configuration
- Providing the final recommendation

The notebook does not contain the core training pipeline.

## Final Model Selection

The final model is selected after completing all experiments.

The recommendation considers:

- Predictive performance
- Training time
- Feature reduction
- Per-class performance
- Confusion matrix
- Computational efficiency

The highest accuracy model is not automatically selected if another configuration provides a better overall trade-off.

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Install required packages:
```bash
pip install -r requirements.txt
```


## Run

From the project root:

```bash
python main.py
```
## Documentation

- `architecture/proposal.md` — project objectives, scope, and experiment strategy
- `architecture/design.md` — architecture, data flow, module responsibilities, and constraints
- `CHANGELOG.md` — project changes
- `HANDOFF.md` — project handoff information
