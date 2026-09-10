from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier


def all_features_fs():
    return SelectKBest(
        score_func=f_classif,
        k="all"
    )


def f_classif_selection(k=350):
    return SelectKBest(
        score_func=f_classif,
        k=k
    )


def mutual_info_selection(k=350):
    return SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )


def rfe_feature_selection(k=350):

    estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    return RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=10
    )
