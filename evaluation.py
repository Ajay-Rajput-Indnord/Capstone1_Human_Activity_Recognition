import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = "results/experiment_results.csv"
OUTPUT_DIR = "results/evaluation"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not os.path.exists(RESULTS_FILE):

        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Please run run_experiment.py first."
        )

    df = pd.read_csv(RESULTS_FILE)

    return df


# ============================================================
# PREPARE RESULTS
# ============================================================

def prepare_results(df):

    # Calculate train-test accuracy gap
    df["accuracy_gap"] = (
        df["train_accuracy"] -
        df["test_accuracy"]
    )

    return df


# ============================================================
# BEST EXPERIMENT
# ============================================================

def evaluate_best_experiment(df):

    best = df.loc[
        df["test_accuracy"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST EXPERIMENT")
    print("=" * 70)

    print(
        "Model              :",
        best["model"]
    )

    print(
        "Feature Selection   :",
        best["feature_selection"]
    )

    print(
        "Number of Features  :",
        best["number_of_features"]
    )

    print(
        "Train Accuracy      :",
        f"{best['train_accuracy']:.4f}"
    )

    print(
        "Test Accuracy       :",
        f"{best['test_accuracy']:.4f}"
    )

    print(
        "Accuracy Gap        :",
        f"{best['accuracy_gap']:.4f}"
    )

    print(
        "Training Time       :",
        f"{best['training_time']:.4f} seconds"
    )

    print(
        "Testing Time        :",
        f"{best['testing_time']:.4f} seconds"
    )

    return best


# ============================================================
# TOP EXPERIMENTS
# ============================================================

def show_top_experiments(df, n=10):

    print("\n" + "=" * 70)
    print(f"TOP {n} EXPERIMENTS")
    print("=" * 70)

    columns = [
        "model",
        "feature_selection",
        "number_of_features",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
        "training_time",
        "testing_time"
    ]

    top = (
        df
        .sort_values(
            by="test_accuracy",
            ascending=False
        )
        .head(n)
    )

    print(
        top[columns].to_string(index=False)
    )

    return top


# ============================================================
# MODEL COMPARISON
# ============================================================

def evaluate_models(df):

    model_summary = (
        df
        .groupby("model")
        .agg(
            average_train_accuracy=(
                "train_accuracy",
                "mean"
            ),

            average_test_accuracy=(
                "test_accuracy",
                "mean"
            ),

            best_test_accuracy=(
                "test_accuracy",
                "max"
            ),

            average_training_time=(
                "training_time",
                "mean"
            ),

            average_testing_time=(
                "testing_time",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "best_test_accuracy",
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        model_summary.to_string(index=False)
    )

    return model_summary


# ============================================================
# FEATURE SELECTION COMPARISON
# ============================================================

def evaluate_feature_selection(df):

    feature_summary = (
        df
        .groupby("feature_selection")
        .agg(
            average_train_accuracy=(
                "train_accuracy",
                "mean"
            ),

            average_test_accuracy=(
                "test_accuracy",
                "mean"
            ),

            best_test_accuracy=(
                "test_accuracy",
                "max"
            ),

            average_training_time=(
                "training_time",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "best_test_accuracy",
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPARISON")
    print("=" * 70)

    print(
        feature_summary.to_string(index=False)
    )

    return feature_summary


# ============================================================
# MODEL + FEATURE MATRIX
# ============================================================

def create_accuracy_matrix(df):

    matrix = pd.pivot_table(
        df,
        values="test_accuracy",
        index="model",
        columns="feature_selection",
        aggfunc="mean"
    )

    print("\n" + "=" * 70)
    print("TEST ACCURACY MATRIX")
    print("=" * 70)

    print(
        matrix.to_string(float_format=lambda x: f"{x:.4f}")
    )

    return matrix


# ============================================================
# PLOT TEST ACCURACY
# ============================================================

def plot_test_accuracy(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    pivot = df.pivot(
        index="model",
        columns="feature_selection",
        values="test_accuracy"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_title(
        "Test Accuracy: Model vs Feature Selection"
    )

    ax.set_xlabel("Model")

    ax.set_ylabel("Test Accuracy")

    ax.set_ylim(0, 1)

    plt.xticks(rotation=45)

    plt.legend(
        title="Feature Selection"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/test_accuracy.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "\nSaved plot:",
        output_file
    )


# ============================================================
# TRAIN VS TEST ACCURACY
# ============================================================

def plot_train_test_accuracy(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    plot_df = df.copy()

    plot_df["experiment"] = (
        plot_df["model"]
        + " + "
        + plot_df["feature_selection"]
    )

    plot_df = plot_df.sort_values(
        "test_accuracy",
        ascending=False
    )

    ax = plot_df.plot(
        x="experiment",
        y=[
            "train_accuracy",
            "test_accuracy"
        ],
        kind="bar",
        figsize=(15, 8)
    )

    ax.set_title(
        "Train Accuracy vs Test Accuracy"
    )

    ax.set_xlabel(
        "Experiment"
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=75,
        ha="right"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/train_vs_test_accuracy.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "\nSaved plot:",
        output_file
    )


# ============================================================
# TRAINING TIME
# ============================================================

def plot_training_time(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    pivot = df.pivot(
        index="model",
        columns="feature_selection",
        values="training_time"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_title(
        "Training Time"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Training Time (seconds)"
    )

    plt.xticks(rotation=45)

    plt.legend(
        title="Feature Selection"
    )

    plt.tight_layout()

    output_file = (
        f"{OUTPUT_DIR}/training_time.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        "\nSaved plot:",
        output_file
    )


# ============================================================
# SAVE SUMMARY TABLES
# ============================================================

def save_evaluation_results(
    df,
    model_summary,
    feature_summary,
    top_experiments
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        f"{OUTPUT_DIR}/all_results.csv",
        index=False
    )

    model_summary.to_csv(
        f"{OUTPUT_DIR}/model_comparison.csv",
        index=False
    )

    feature_summary.to_csv(
        f"{OUTPUT_DIR}/feature_selection_comparison.csv",
        index=False
    )

    top_experiments.to_csv(
        f"{OUTPUT_DIR}/top_experiments.csv",
        index=False
    )

    print("\nEvaluation CSV files saved to:")
    print(OUTPUT_DIR)


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluation():

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Load
    df = load_results()

    print(
        f"\nLoaded {len(df)} experiments."
    )

    # Prepare
    df = prepare_results(df)

    # Best experiment
    evaluate_best_experiment(df)

    # Top experiments
    top_experiments = show_top_experiments(
        df,
        n=10
    )

    # Model comparison
    model_summary = evaluate_models(df)

    # Feature selection comparison
    feature_summary = evaluate_feature_selection(df)

    # Accuracy matrix
    create_accuracy_matrix(df)

    # Plots
    plot_test_accuracy(df)

    plot_train_test_accuracy(df)

    plot_training_time(df)

    # Save
    save_evaluation_results(
        df,
        model_summary,
        feature_summary,
        top_experiments
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


# if __name__ == "__main__":
#     main()
