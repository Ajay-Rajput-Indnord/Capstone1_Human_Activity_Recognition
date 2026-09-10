import pandas as pd

def load_data():
    path=r'dataset\.csv'
    global data
    data=pd.read_csv(path)
    return data


def test_data():
    test_path=r'dataset\test.csv'
    global data
    test_data=pd.read_csv(test_path)
    return test_data


def data_info():
    shape=data.shape
    print(shape)
    info=data.info
    print(info)
    df=data.head(5)
    print(df)

def feture():
    global X
    X = load_data().drop(columns=["subject", "Activity"])
    return X
def target():
    global y
    y = load_data()["Activity"]
    return y

def test_feture():
    global test_X
    test_X = test_data().drop(columns=["subject", "Activity"])
    return test_X
def test_target():
    global test_y
    test_y = test_data()["Activity"]

    return test_y



    

