#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from src.feature_selection import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from src.data_loader import feture, target
from src.data_loader import test_feture, test_target
from src.feature_selection import rfe_feature_selection
import time


# =========================================================
# extra_trees - ENSEMBLE FAMILY
# =========================================================
def extra_trees_model():
    
    extra_trees_model = Pipeline([
           ("transform", PowerTransformer(method="yeo-johnson")),
    ("selection",
        rfe_feature_selection(k=350)),
        #SelectKBest(f_classif, k=350)),
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
    

    return extra_trees_model

# =========================================================
# logistic_regression - Linear / statistical familyY
# =========================================================
def logistic_regression_model():
    logistic_regression_model= Pipeline([
        ("scaler",StandardScaler()),
        ("select", 
            rfe_feature_selection(k=350)),
         #SelectKBest(f_classif, k=250)),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
          #  multi_class="auto",
            random_state=42
        ))
    ])


    return logistic_regression_model

# =========================================================
# KNN - Instance-based family
# =========================================================
def knn_model():
    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("select",rfe_feature_selection(k=350)),
 
         #SelectKBest(f_classif, k=150)),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ))
    ])
   
    return knn_model


# =========================================================
# rbf_svm - Kernal FAMILY
# =========================================================
def rbf_svm_model():
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("select", 
            rfe_feature_selection(k=350)),

         #SelectKBest(f_classif, k=150)),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        ))
    ])
  

    return svm



# =========================================================
# RANDOM FOREST - ENSEMBLE FAMILY
# =========================================================

def random_forest_model():

    random_forest = Pipeline([
        ("scaler", StandardScaler()),

        ("select",
            rfe_feature_selection(k=350)),
            #SelectKBest(score_func=f_classif,k=350)),

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


    return random_forest


# =========================================================
# LDA - DISCRIMINANT FAMILY
# =========================================================

def lda_model():

    lda = Pipeline([
        ("scaler", StandardScaler()),

        ("select",
         rfe_feature_selection(k=350)),
    #     SelectKBest(score_func=mutual_info_classif,k=350)),

        ("model", LinearDiscriminantAnalysis(
            solver="svd"
        ))
    ])


    return lda

from sklearn.ensemble import ExtraTreesClassifier
def model_prediction():
    # TAKE input x_train and y_train select model and fit it on this data and predict
    X_train=feture()
    y_train=target()
    global model
    model=logistic_regression_model()
    start = time.time()
    fitted_model=model.fit(X_train, y_train)
    end = time.time()
    print("model traing time is :", end - start)
    val_pred = fitted_model.predict(X_train)
    
    return  model, val_pred, y_train

def model_pred_test():
    # use test data and pridict output
    #model, val_pred, y_train=model_prediction()
    tf=test_feture()
    tt=test_target()   
    test_output=model.predict(tf)   
    return test_output
