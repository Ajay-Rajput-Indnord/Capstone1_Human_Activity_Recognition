import pandas as pd

def load_data():
    path=r'dataset\train.csv'
    global data
    data=pd.read_csv(path)
    return data

    #print('data loaded sucsses full')


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
    

