import os
import time
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import feture, target
from src.data_loader import test_feture, test_target

from src.models import (
    extra_trees_model,
    logistic_regression_model,
    knn_model,
    rbf_svm_model,
    random_forest_model,
    lda_model
)

from src.feature_selection import (
    all_features_fs,
    f_classif_selection,
    rfe_feature_selection
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Extra Trees": extra_trees_model,
    "Logistic Regression": logistic_regression_model,
    "KNN": knn_model,
    "RBF SVM": rbf_svm_model,
    "Random Forest": random_forest_model,
    "LDA": lda_model
}


# ============================================================
# FEATURE SELECTION METHODS
# ============================================================

feature_methods = {
    "all_features": all_features_fs,
    "F-Classif": f_classif_selection,
    "RFE": rfe_feature_selection
}


# Number of selected features
N_FEATURES = 350


# ============================================================
# RUN EXPERIMENTS
# ============================================================

def run_all_experiments():

    print("Loading dataset...")

    X_train = feture()
    y_train = target()

    X_test = test_feture()
    y_test = test_target()

    print("Training samples:", X_train.shape[0])
    print("Training features:", X_train.shape[1])

    print("Testing samples:", X_test.shape[0])
    print("Testing features:", X_test.shape[1])

    results = []

    total_experiments = len(models) * len(feature_methods)
    experiment_number = 0

    for model_name, model_function in models.items():

        for feature_name, feature_function in feature_methods.items():

            experiment_number += 1

            print("\n" + "=" * 70)
            print(
                f"EXPERIMENT {experiment_number}/{total_experiments}"
            )
            print("MODEL:", model_name)
            print("FEATURE SELECTION:", feature_name)
            print("=" * 70)

            # ------------------------------------------------
            # Create fresh feature selector
            # ------------------------------------------------

            selector = feature_function(k=N_FEATURES)

            # ------------------------------------------------
            # Create fresh model
            # ------------------------------------------------

            model = model_function(selector)

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            print("Training...")

            start_time = time.time()

            model.fit(X_train, y_train)

            training_time = time.time() - start_time

            # ------------------------------------------------
            # TRAIN PREDICTION
            # ------------------------------------------------

            train_start = time.time()

            train_pred = model.predict(X_train)

            train_prediction_time = time.time() - train_start

            # ------------------------------------------------
            # TEST PREDICTION
            # ------------------------------------------------

            test_start = time.time()

            test_pred = model.predict(X_test)

            testing_time = time.time() - test_start

            # ------------------------------------------------
            # ACCURACY
            # ------------------------------------------------

            train_accuracy = accuracy_score(
                y_train,
                train_pred
            )

            test_accuracy = accuracy_score(
                y_test,
                test_pred
            )

            # ------------------------------------------------
            # CLASSIFICATION REPORT
            # ------------------------------------------------

            train_report = classification_report(
                y_train,
                train_pred,
                zero_division=0
            )

            test_report = classification_report(
                y_test,
                test_pred,
                zero_division=0
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            results.append({

                "model": model_name,

                "feature_selection": feature_name,

                "number_of_features": N_FEATURES,

                "train_accuracy": train_accuracy,

                "test_accuracy": test_accuracy,

                "training_time": training_time,

                "train_prediction_time": train_prediction_time,

                "testing_time": testing_time,

                "train_classification_report": train_report,

                "test_classification_report": test_report
            })

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print(
                f"Train Accuracy : {train_accuracy:.4f}"
            )

            print(
                f"Test Accuracy  : {test_accuracy:.4f}"
            )

            print(
                f"Training Time  : {training_time:.4f} seconds"
            )

            print(
                f"Testing Time   : {testing_time:.4f} seconds"
            )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    output_file = "results/experiment_results.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")
    print(output_file)

    print("\nTotal experiments:", len(df))


# ============================================================
# MAIN
# ============================================================

# if __name__ == "__main__":

#     results = run_all_experiments()

#     save_results(results)
