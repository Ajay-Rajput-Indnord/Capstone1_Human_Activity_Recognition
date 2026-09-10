from sklearn.metrics import accuracy_score, classification_report
from src.models import model_prediction, model_pred_test
from src.data_loader import test_feture, test_target
import json




def traing():
    model, val_pred, y_train=model_prediction()
    train_acc=accuracy_score(y_train, val_pred)
    
    train_report =classification_report(y_train, val_pred)
    
    

    return train_acc, train_report
    



def testing():
    test_prediction = model_pred_test()
    tg=test_target()
    test_acc=accuracy_score(tg,test_prediction)
    
    test_report =classification_report(tg,test_prediction)
    return test_acc, test_report

def create_result(file_name):
    train_acc, train_report=traing()
    test_acc, test_report=testing()
    
    results = {
    "train_accuracy": train_acc,
    "train_classification_report": train_report,
    "test_accuracy": test_acc,
    "test_classification_report": test_report
    }
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)

    print("results Saved to ", file_name)


