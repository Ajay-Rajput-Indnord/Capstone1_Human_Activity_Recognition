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
    #all_features_fs,
    f_classif_selection,
    mutual_info_selection,
    rfe_feature_selection
)


# models 
models = {
    "Extra Trees": extra_trees_model,
    "Logistic Regression": logistic_regression_model,
    "KNN": knn_model,
    "RBF SVM": rbf_svm_model,
    "Random Forest": random_forest_model,
    "LDA": lda_model
}



feature_methods = {
    #"all_features": all_features_fs,
    "F-Classif": f_classif_selection,
    "Mutual Information": mutual_info_selection,
    "RFE": rfe_feature_selection
}


def run_all_experiments():

    X_train = feture()
    y_train = target()

    X_test = test_feture()
    y_test = test_target()

    results = []

    for model_name, model_function in models.items():

        for feature_name, feature_function in feature_methods.items():

            print("\n" + "=" * 70)
            print("MODEL:", model_name)
            print("FEATURE SELECTION:", feature_name)
            print("=" * 70)

            # Create a fresh selector
            selector = feature_function(k=350)

            # Create a fresh model
            model = model_function(selector)

            # -------------------------
            # TRAIN
            # -------------------------

            start = time.time()

            model.fit(X_train, y_train)

            train_time = time.time() - start

            # -------------------------
            # TRAIN PREDICTION
            # -------------------------

            train_pred = model.predict(X_train)

            # -------------------------
            # TEST PREDICTION
            # -------------------------

            start = time.time()

            test_pred = model.predict(X_test)

            test_time = time.time() - start

            # -------------------------
            # ACCURACY
            # -------------------------

            train_accuracy = accuracy_score(
                y_train,
                train_pred
            )

            test_accuracy = accuracy_score(
                y_test,
                test_pred
            )

            # -------------------------
            # REPORT
            # -------------------------

            train_report = classification_report(
                y_train,
                train_pred
            )

            test_report = classification_report(
                y_test,
                test_pred
            )

            # -------------------------
            # SAVE ONE ROW
            # -------------------------
            results.append({
                "model": model_name,
                "feature_selection": feature_name,
                "number_of_features": 350,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "training_time": train_time,
                "testing_time": test_time,
                "train_classification_report": train_report,
                "test_classification_report": test_report
            })

            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy :", test_accuracy)

    return results


def save_results(results):

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(
        "results/experiment_results.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/experiment_results.csv")


if __name__ == "__main__":

    results = run_all_experiments()

    save_results(results)
