from data_loader import feture, target
import pandas as pd
# x=feture()
# print(x)
# y=target()
# print(y)

from sklearn.feature_selection import SelectKBest, f_classif

def f_classif_fs():
    
    selector = SelectKBest(f_classif, k=150)
    return selector


from sklearn.feature_selection import SelectKBest, mutual_info_classif
def mutual_info_classif_fs():
    
    selector = SelectKBest(mutual_info_classif, k=150)
    return selector

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
def ExtraTreesClassifier_fc():
    
    selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        max_features=150
    )
    return selector
