from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC




# =========================================================
# extra_trees - ENSEMBLE FAMILY
# =========================================================
def extra_trees_model(selector):
    
    return  Pipeline([
           ("transform", PowerTransformer(method="yeo-johnson")),
    ("selection",selector),
    ("classifier", ExtraTreesClassifier(
        n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
    ))
])
    



# =========================================================
# logistic_regression - Linear / statistical familyY
# =========================================================
def logistic_regression_model(selector):
    return Pipeline([
        ("scaler",StandardScaler()),
        ("select", selector),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
          #  multi_class="auto",
            random_state=42
        ))
    ])



# =========================================================
# KNN - Instance-based family
# =========================================================
def knn_model(selector):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", selector),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ))
    ])
   


# =========================================================
# rbf_svm - Kernal FAMILY
# =========================================================
def rbf_svm_model(selector):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", selector),
            
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        ))
    ])
  

  



# =========================================================
# RANDOM FOREST - ENSEMBLE FAMILY
# =========================================================

def random_forest_model(selector):

    return Pipeline([
        ("scaler", StandardScaler()),

        ("select", selector),

        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])




# =========================================================
# LDA - DISCRIMINANT FAMILY
# =========================================================

def lda_model(selector):

    return Pipeline([
        ("scaler", StandardScaler()),

        ("select", selector),

        ("model", LinearDiscriminantAnalysis(
            solver="svd"
        ))
    ])


