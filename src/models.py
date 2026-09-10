#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.feature_selection import ExtraTreesClassifier
from sklearn.svm import SVC
from src.data_loader import feture, target
from src.data_loader import test_feture, test_target
import time


def extra_trees_model():
    start=time.time()
    extra_trees_model = Pipeline([
           ("transform", PowerTransformer(method="yeo-johnson")),
    ("selection", SelectKBest(f_classif, k=350)),
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
    end= time.time()
    print('model training time is :', start - end)

    return extra_trees_model

def logistic_regression_model():
    start=time.time()
    logistic_regression_model= Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=250)),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
          #  multi_class="auto",
            random_state=42
        ))
    ])
    end= time.time()
    print('model training time is :', start - end)

    return logistic_regression_model

def knn_model():
    start=time.time()
    knn_model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=150)),
        ("model", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ))
    ])
    end= time.time()
    print('model training time is :', start - end)
    return knn_model

def rbf_svm_model():
    start=time.time()
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=150)),
        ("model", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            random_state=42
        ))
    ])
    end= time.time()
    print('model training time is :', start - end)

    return svm



from sklearn.ensemble import ExtraTreesClassifier
def model_prediction():
    X_train=feture()
    y_train=target()
    model=logistic_regression_model()
    fitted_model=model.fit(X_train, y_train)
    val_pred = fitted_model.predict(X_train)
    
    return  model, val_pred, y_train

def model_pred_test():
    model, val_pred, y_train=model_prediction()
    tf=test_feture()
    tt=test_target()   
    test_output=model.predict(tf)   
    return test_output
