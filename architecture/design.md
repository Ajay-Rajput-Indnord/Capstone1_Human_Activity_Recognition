# Design: Human Activity Recognition

## Overview

Implement a modular Machine Learning architecture for **Human Activity Recognition (HAR)** using smartphone sensor data.

The system will evaluate all selected classifiers across three feature conditions:
1. **All Features** — all 561 sensor-derived features.
2. **Mutual Information** — filter-based feature selection.
3. **RFE** — wrapper-based Recursive Feature Elimination.

The architecture separates dataset handling, preprocessing, feature selection, model construction, training, evaluation, result storage, and reporting.

The core Machine Learning pipeline will be implemented independently from the final reporting notebook.

---
## Architectural Decisions

### Data Source

Use the existing `train.csv` and `test.csv` datasets as the primary data sources.

Reason: The project already provides predefined training and testing datasets, allowing the models to be evaluated consistently using the same data split.

The original dataset files must remain unchanged.

---

### Feature and Target Separation

The dataset will be divided into:

- **Sensor Features:** 561 feature columns.
- **Subject:** Participant identifier.
- **Activity:** Target variable.

The `subject` column will not be included as a normal model feature because it identifies the participant rather than representing sensor measurements.

The `Activity` column will be used as the classification target.

---

### Feature Conditions

Three feature conditions will be implemented.

Reason: Comparing the complete feature space with two different feature-selection methodologies allows the effect of dimensionality reduction to be evaluated systematically.

---
### Feature Selection — Mutual Information

Use **Mutual Information** as the filter-based feature-selection technique.

The selector will be fitted using training data only and the same fitted selector will transform both training and test features.

---
### Feature Selection — RFE

Use **Recursive Feature Elimination (RFE)** as the wrapper-based feature-selection technique.

RFE will use a configured estimator to recursively remove less important features. The selector will be fitted using training data only and then applied to the test features.

The exact RFE estimator and number of selected features will be maintained as configurable parameters.

---

### Classifier Architecture

All selected classifiers will be evaluated rather than selecting one classifier before experimentation.

The classifiers will be organized by Machine Learning family:

```text
Classifiers
    │
    ├── Linear-Based
    │
    ├── Tree-Based
    │
    ├── Kernel-Based
    │
    ├── Ensemble
    │
    └── Discriminant
```

Each classifier must expose a consistent `fit()` and `predict()` interface through the selected Machine Learning framework.

---
### Experiment Matrix

Every configured classifier will be evaluated under all three feature conditions.

N classifiers × 3 feature conditions = N × 3 experiments

This ensures that every classifier receives the same feature-condition comparison.

---

## Component & Data Flow

The system will be divided into the following logical components:

```text
                    ┌──────────────────┐
                    │   Dataset Files  │
                    │ train.csv/test   │
                    │   dataset/       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Data Loader &   │
                    │    Validator     │
                    │ data_loader.py   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature / Target │
                    │    Processor     │
                    │     main.py      │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌───────────┐   ┌──────────┐
        │   ALL    │   │  Mutual   │   │   RFE    │
        │ Features │   │   Info    │   │          │
        │          │   │           │   │          │
        └────┬─────┘   └─────┬─────┘   └────┬─────┘
             │               │              │
             └───────────────┼──────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Classifier    │
                    │     models.py    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Experiment Runner│
                    │run_experiment.py │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Evaluation    │
                    │  evaluation.py   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Result Storage  │
                    │results/          │
                    │experiment_results│
                    │     .csv         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Reporting in     │
                    │    Notebook      │
                    │results/          │
                    │ notebook.ipynb   │
                    └──────────────────┘
```

---

## Component Responsibilities

### 1. Data Loader

Responsible for loading the training and testing datasets.

---

### 2. Data Validator

Responsible for checking:

- Required columns.
- Missing values.
- Feature counts.
- Target availability.
- Training/test feature consistency.
- Data types where required.

---

### 3. Feature Processor

Responsible for:

- Separating sensor features from metadata.
- Separating `subject`.
- Separating `Activity`.
- Preparing the feature matrices used by the experiments.

---

### 4. Feature Selection Module

The feature-selection module will provide a common interface for the different feature conditions.

Supported methods:

```text
all
mutual_information
rfe
```

For the `all` condition, no feature selection will be performed.

For `mutual_information`, the Mutual Information selector will be created.

For `rfe`, the configured RFE selector will be created.

---

### 5. Classifier Factory

The classifier factory will centralize model construction.

The factory will return a configured classifier instance.

Classifier configuration should include relevant parameters such as:

```text
classifier name
classifier family
hyperparameters
random state where applicable
```

Keeping classifier creation centralized prevents different experiments from accidentally using inconsistent configurations.

---

### 6. Experiment Runner

The experiment runner is responsible for orchestrating the complete experiment matrix.

The runner will:

1. Load the dataset.
2. Validate the dataset.
3. Prepare training and testing features.
4. Generate the three feature conditions.
5. Iterate through every configured classifier.
6. Train each classifier.
7. Measure training time.
8. Generate predictions.
9. Calculate evaluation metrics.
10. Store the results.

---

## Evaluation Component

The evaluation component will provide a common interface for all experiments.

The evaluator will calculate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The evaluation procedure must remain consistent across all classifier-feature combinations.

---

## Training-Time Measurement

Training time will be measured around the classifier training operation.

Conceptual flow:

```text
Start Timer
     ↓
model.fit(X_train, y_train)
     ↓
Stop Timer
     ↓
Store Training Time
```

Training time will be recorded separately for every experiment.

Feature-selection time may also be recorded separately if required for computational analysis.

---

## Result Schema

The result storage layer will maintain a consistent structure.

Minimum result fields:

```text
classifier
classifier_family
feature_condition
accuracy
precision
recall
f1_score
training_time
```

Results will be saved to:

```text
results/
└── experiment_results.csv
```

---

## Confusion Matrix Storage

A confusion matrix will be generated for every classifier-feature combination and included in the final report.

---

## Configuration Management

Model and experiment configurations will be separated from execution logic.

A centralized configuration should define:

```text
Feature Conditions
Classifier List
Classifier Families
Mutual information Parameters
RFE Parameters
Evaluation Parameters
Output Paths
Random States
```

This allows experiments to be modified without changing the core experiment runner.

---

## Proposed Project Structure

The architecture should follow a modular structure similar to:

```text
Capstone_1/
│
├── architecture/
│   ├── design.md
│   └── proposal.md
│
├── dataset/
│   ├── test.csv
│   └── train.csv
│
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── main.py
│   └── models.py
│
├── run_experiment.py
|
├── results/
|   |── experiment_results.csv
│   └── notebook.ipynb
│
├── .gitignore
├── CHANGELOG.md
├── HANDOFF.md
├── README.md
├── requirements.txt

```
---

## Security & Data Integrity Considerations

### Test Data Isolation

The test dataset must remain isolated from model fitting.

The following operations must not use test data:

- Feature-selection fitting
- Model training
- Hyperparameter selection based on test performance

The test dataset will only be used for final evaluation.

### Dataset Integrity

The original dataset files will be treated as read-only inputs.

No preprocessing operation should overwrite the original CSV files.

### Reproducibility

Randomized algorithms should use fixed random states where supported.

Experiment configurations should be version-controlled alongside the project.

---

## Leakage Prevention

The architecture must ensure that feature selection occurs only on training data.

Feature selection must be fitted exclusively on training data. The fitted selector is then applied to both training and test data. Test data must never influence feature selection or model training.

---

## Experiment Reproducibility

The complete experiment will be executable using:

```bash
python run_experiment.py
```

The command will execute the configured classifier × feature-condition matrix and generate the experiment results.

The reporting notebook will read the persisted results rather than retraining models unnecessarily.

---

## Reporting Architecture

The reporting notebook will remain separate from the core Machine Learning implementation.

```text
                 Core Pipeline
                      │
                      ▼
              experiment_results.csv
                      │
                      ▼
                Notebook.ipynb
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Comparisons  Charts     Analysis
```

The notebook will be responsible for:

- Loading results.
- Comparing classifiers.
- Comparing feature conditions.
- Visualizing accuracy.
- Visualizing training time.
- Displaying confusion matrices.
- Analyzing per-class metrics.
- Identifying the best-performing configuration.
- Presenting the final recommendation.

---

## Final Model Selection

No classifier will be considered the final model before the experiments are completed.

The final recommendation will be based on predictive performance, computational efficiency, feature reduction, and per-class performance. The highest accuracy model will not automatically be selected if another configuration provides a better overall trade-off.

---

## Architectural Constraints

The following constraints are mandatory:

- All selected classifiers must be evaluated.
- Every classifier must be evaluated under all three feature conditions.
- The three feature conditions must be All Features, Mutual information and RFE.
- Mutual information must be implemented as the filter-based selection method.
- RFE must be implemented as the wrapper-based selection method.
- Test data must not be used for feature selection or model training.
- The `subject` identifier must remain separate from the sensor feature matrix.
- Original dataset files must not be modified.
- Training time must be recorded.
- Accuracy, precision, recall, F1-score, and confusion matrices must be generated.
- Experiment results must be persisted.
- Model configurations must be centralized.
- The experiment must be executable through a single entry point.
- Core Machine Learning logic must remain outside the final reporting notebook.
- The architecture must remain modular and reproducible.

---

## Implementation Sequence

The recommended implementation order is:

```text
1. Dataset Loader
        ↓
2. Dataset Validator
        ↓
3. Feature / Target Processor
        ↓
4. Mutual information Feature Selector
        ↓
5. RFE Feature Selector
        ↓
6. Classifier Factory
        ↓
7. Evaluation Module
        ↓
8. Experiment Runner
        ↓
9. Result Storage
        ↓
10. Final Reporting Notebook
```

Each component should be independently testable before being integrated into the complete experiment pipeline.

---