from src.data_loader import feture, target
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def all_features_fs():
    # to select all feature using f_classif
    selector = SelectKBest(
        score_func=f_classif,
        k="all"
    )
    return selector


def f_classif_fs():
    # selecting top 150 feature using f_classif
    
    selector = SelectKBest(f_classif, k=150)
    return selector


def mutual_info_classif_fs():
    # selecting top 150 feature using mutual_info_classif

    
    selector = SelectKBest(mutual_info_classif, k=150)
    return selector






def rfe_feature_selection(k=350):
    # selecting top 350 feature using rfe_feature_selection


    estimator = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=k,
        step=10
    )

    return selector




