from src.evaluation import traing, testing, create_result




def main():
    train_acc, train_report=traing()
    test_acc, test_report=testing()
    
    print(f"model performance in traning  is {train_acc} accuracy \n model performance in traing  is \n {train_report} \n model performance in testing  is \n {test_acc} \n model performance in testing is \n  {test_report}")
    #give file name
    create_result(r'results\f_classif\logistic_regression_model.json')
    


if __name__ == "__main__":
    main()