import pandas as pd

def load_data():
    # use to load train dataset input is train data path
    #return train data
    path=r'dataset\train.csv'
    data=pd.read_csv(path)
    return data


def test_data():
    # use to load test dataset input is test data path
    # return test data
    test_path=r'dataset\test.csv'
    test_data=pd.read_csv(test_path)
    return test_data


def data_info():
    # give information about train data set loke shape of data
    data = load_data()
    shape=data.shape
    print(shape)
    info=data.info
    print(info)
    df=data.head(5)
    print(df)

def feture():
    # take input data and extract feture for model training 
    X = load_data().drop(columns=["subject", "Activity"])
    return X
def target():
    # take input data and extract target for model training 

    y = load_data()["Activity"]
    return y

def test_feture():
    # testing data fetures extraction
    test_X = test_data().drop(columns=["subject", "Activity"])
    return test_X
def test_target():
    # testing data target extraction
    test_y = test_data()["Activity"]

    return test_y



    

