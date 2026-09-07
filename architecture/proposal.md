# Proposal: Human Activity Recognition

## 1. Summary

Human Activity Recognition (HAR) is a machine learning classification project that uses smartphone sensor data to automatically identify human activities. The project uses the Dataset containing 561 sensor features collected from smartphone accelerometer and gyroscope measurements.

The system will classify observations into six human activities.

The project will compare multiple classifiers from different Machine Learning families and evaluate their performance under three feature conditions:

1. All 561 sensor-derived features
2. Features selected using **Mutual Information**
3. Features selected using **Recursive Feature Elimination (RFE)**

Data loading, preprocessing, feature selection, model training, evaluation, and experiment execution will be organized into separate modules. The final notebook will be used only for reporting and visualization of results generated during the process.

---

## 2. Problem Statement

The Human Activity Recognition dataset contains 561 sensor-derived features for each observation. The high-dimensional feature space may affect classifier performance, training time, and computational requirements.

Different Machine Learning algorithms may behave differently on the same sensor feature space. Therefore, selecting a classifier without systematic experimentation may result as a poor model choice.

A controlled comparison of multiple classifiers from different algorithm families is required to determine which approaches perform best for Human Activity Recognition.

The project will also investigate whether feature selection can reduce the dimensionality of the dataset while maintaining or improving classification performance and reducing computational cost.
---

## 3. Scope of Work

### What is Changing (In-Scope)

- **ADDED**: A modular data-loading and validation component for train.csv and test.csv.
- **ADDED**: An all-features experimental condition using the complete sensor feature set.
- **ADDED**: A filter-based feature-selection condition using **Mutual Information**.
- **ADDED**: A wrapper-based feature-selection condition using **Recursive Feature Elimination (RFE)**.
- **ADDED**: The classifiers will span multiple Machine Learning families.
- **ADDED**: A common experiment pipeline for evaluating every classifier under every feature condition.
- **ADDED**: Evaluation of each classifier-feature combination using accuracy, per-class metrics, confusion matrices, and training time.
- **ADDED**: Persistent storage of experiment results in a machine-readable format.
- **ADDED**: A final reporting notebook that shows results.

### What is Not Changing (Out-of-Scope)

- We are not implementing real-time sensor acquisition.
- We are not modifying the original train.csv or test.csv datasets.
- We are not using the test dataset for model training.
- We are not using test data to fit feature-selection techniques.
- We are not implementing Deep Learning models as part of this experiment.

---

## 4. Experiment Strategy

The experiment will evaluate every selected classifier under three feature conditions:

1. **All Features** — All 561 sensor-derived features.
2. **Mutual Information** — Filter-based feature selection using Mutual Information.
3. **RFE** — Wrapper-based feature selection.

The selected classifiers will represent multiple Machine Learning algorithm families. Each classifier will be evaluated under all three feature conditions using the same evaluation procedure to ensure a fair comparison.

---

## 5. Data Processing Strategy

The dataset contains:

- **561 sensor-derived features**
- `subject` identifier
- `Activity` target

The Activity column will be used as the target variable. The subject column will be kept separate from the sensor feature matrix because it identifies the participant rather than representing a sensor measurement.

The data will be validated and separated into features, target, and subject information before applying the different feature conditions and training the classifiers.

The original datasets will remain unchanged throughout the experiment.

---

## 6. Feature Selection Strategy

Two feature-selection techniques from different methodological families will be evaluated in addition to the all-features condition.

### 6.1 Mutual Information

Mutual Information will be used as a filter-based feature-selection method to identify features that provide relevant information about the target activity classes.

### 6.2 Recursive Feature Elimination (RFE)

Recursive Feature Elimination (RFE) will be used as a wrapper-based feature-selection method to identify a relevant subset of features.

### 6.3 Feature Conditions

The experiment will compare three feature conditions:

1. **All Features** — All 561 sensor-derived features.
2. **Mutual Information** — Features selected using the filter-based Mutual Information method.
3. **RFE** — Features selected using the wrapper-based Recursive Feature Elimination method.

Feature selection will be performed using training data only. The test dataset will not be used during feature selection to prevent information leakage.

## 7. Classifier Strategy

The project will evaluate all selected classifiers rather than selecting a single algorithm in advance.

The classifiers will span multiple Machine Learning families, including:

- Linear-based classifiers
- Tree-based classifiers
- Kernel-based classifiers
- Ensemble classifiers
- Discriminant classifiers

The exact classifier configurations will be defined in design.md.

Each classifier will be evaluated under all three feature conditions

The final classifier will be selected based on a combination of:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training time
- Effect of feature selection

---

## 8. Evaluation Strategy

Each classifier-feature combination will be evaluated using accuracy, precision, recall, F1-score, confusion matrices, and training time. The final recommendation will consider predictive performance and computational cost.

---

## 9. Reporting Strategy

The final notebook will act as the reporting and visualization layer rather than the main implementation environment.

The notebook will present experiment configuration, model comparisons, performance metrics, confusion matrices, computational cost, feature-selection effects, and the final model recommendation.

Core data processing, model training, feature selection, and evaluation logic will remain outside the notebook.

---

## 10. Risks & Dependencies

- **Risk**: Data leakage during feature selection could produce overly optimistic evaluation results.
  - **Mitigation**: Fit Mutual Information and RFE using training data only and prevent test data from influencing feature selection.

- **Risk**: The 561-dimensional feature space may increase training time for certain classifiers.
  - **Mitigation**: Compare all-feature performance against Mutual information and RFE conditions and record training time.

- **Risk**: RFE may require significant computational resources because it repeatedly trains an estimator.
  - **Mitigation**: Record feature-selection execution time and configure the number of selected features appropriately.

- **Risk**: Different classifiers may have different computational requirements.
  - **Mitigation**: Measure training time consistently across all experiments.

- **Risk**: Similar activities may be difficult to distinguish.
  - **Mitigation**: Use per-class metrics and confusion matrices to identify common misclassification patterns.

- **Risk**: The subject identifier may introduce subject-specific information if treated as a normal feature.
  - **Mitigation**: Keep the identifier separate from the sensor feature matrix.

- **Dependency**: The experiment depends on the provided train.csv and test.csv datasets.
  - **Mitigation**: Validate the datasets before running the experiments.

---

## 11. Expected Outcome

The proposed architecture will provide a reproducible framework for systematically comparing the selected Machine Learning classifiers across three feature conditions:

The experiment will identify the strongest classifier-feature combination based on predictive performance and computational efficiency.

The resulting analysis will provide insight into:

- The effect of Mutual Information-based feature selection on model accuracy.
- The effect of RFE-based feature selection on model accuracy.
- The effect of feature selection on training time.
- Differences between classifier families.
- Activities that are most frequently confused.
- The trade-off between model performance and computational cost.

The final model recommendation will be supported by experimental evidence rather than theoretical assumptions.

---