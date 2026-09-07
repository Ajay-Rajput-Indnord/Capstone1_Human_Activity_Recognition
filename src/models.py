from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from src.data_loader import feture, target




def train_test_split_fun():
    X=feture()
    y=target()
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
    return X_train, X_test, y_train, y_test






def extra_trees_model():
    extra_trees_model = Pipeline([
    ("selection", SelectKBest(f_classif, k=150)),
    ("classifier", ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

    return extra_trees_model





from sklearn.ensemble import ExtraTreesClassifier
def model_prediction():
    X_train, X_test, y_train, y_test = train_test_split_fun()
    model=extra_trees_model()
    mod=model.fit(X_train, y_train)
    extra_pred = mod.predict(X_test)
    return extra_pred
    