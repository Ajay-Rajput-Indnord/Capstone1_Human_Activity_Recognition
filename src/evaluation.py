from sklearn.metrics import accuracy_score, classification_report
from src.models import model_prediction, model_pred_test
from src.data_loader import test_feture, test_target
import json




def traing():
    # use model prediction and do evaluation on train data set
    model, val_pred, y_train=model_prediction()
    train_acc=accuracy_score(y_train, val_pred)
    
    train_report =classification_report(y_train, val_pred)
    
    

    return train_acc, train_report
    



def testing():
    #use modelprediction on test data and do evaluation  of model on it 
    test_prediction = model_pred_test()
    tg=test_target()
    test_acc=accuracy_score(tg,test_prediction)
    
    test_report =classification_report(tg,test_prediction)
    return test_acc, test_report




def create_result(file_name):
    # creat filr to store model results

    train_acc, train_report = traing()
    test_acc, test_report = testing()

    results = {
        "train_accuracy": train_acc,
        "train_classification_report": train_report,
        "test_accuracy": test_acc,
        "test_classification_report": test_report
    }

    with open(file_name, "w") as f:
        f.write(f"{train_acc}\n\n")
        f.write("model performance in training is\n")
        f.write(train_report)
        f.write("\n\n")

        f.write(f"{test_acc}\n\n")
        f.write("model performance in testing is\n")
        f.write(test_report)

    print("Results saved to:", file_name)

