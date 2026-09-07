from sklearn.metrics import accuracy_score, classification_report
from src.models import train_test_split_fun, model_prediction



def accuracy():
    X_train, X_test, y_train, y_test = train_test_split_fun()
    extra_pred=model_prediction()
    


    print("Extra Trees Accuracy:", accuracy_score(y_test, model_prediction()))


def report():
    X_train, X_test, y_train, y_test = train_test_split_fun()
    extra_pred=model_prediction()
    print("classification_report", classification_report(y_test, model_prediction()))
