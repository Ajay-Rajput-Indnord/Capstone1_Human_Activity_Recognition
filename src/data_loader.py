import pandas as pd

def load_data(path):
    global data
    data=pd.read_csv(path)

    print('data loaded sucsses full')


def data_info():
    shape=data.shape
    print(shape)
    info=data.info
    print(info)
    df=data.head(5)
    print(df)