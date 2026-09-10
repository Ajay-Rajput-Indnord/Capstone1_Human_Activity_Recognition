from src.data_loader import feture, target
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def all_features_fs():
    selector = SelectKBest(
        score_func=f_classif,
        k="all"
    )
    return selector


def f_classif_fs():
    
    selector = SelectKBest(f_classif, k=150)
    return selector


def mutual_info_classif_fs():
    
    selector = SelectKBest(mutual_info_classif, k=150)
    return selector


def ExtraTreesClassifier_fc():
    
    selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        max_features=150
    )
    return selector
