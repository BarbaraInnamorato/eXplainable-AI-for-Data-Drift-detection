import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skmultiflow.data.data_stream import DataStream
from sklearn.preprocessing import LabelEncoder
from .drift_injection import inject_drift
matplotlib.use('Agg')


def read_data_electricity_market(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "elecNormNew.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    # Pearson correlation
    # plt.figure(figsize=(12, 10))
    # cor = X.corr()
    # pd_cor = pd.DataFrame(cor)
    # pd_cor.to_excel('other_files/'+ f'CORR_elecNormNew.xlsx')
    # #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.title('pearson correlation for electricity dataset')
    # plt.savefig('images/pearson correlation for electricity dataset')

    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    return X, y


def read_data_weather(foldername="data/weather/", shuffle=False):
    df_labels = pd.read_csv(foldername + "NEweather_class.csv", header=None)

    y = df_labels.values.flatten()  # numpy ndarray
    # y.columns = ['PRCP']

    df_data = pd.read_csv(foldername + "NEweather_data.csv", header=None)

    df = df_data.copy()
    df['y'] = y
    df.columns = ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP', 'MXSPD', 'MAX', 'MIN', 'PRC']

    if shuffle is True:
        df = df.sample(frac=1)

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1:]  # 0,1

    # sns.countplot(df['PRC'])

    # Add labels
    # plt.title('Countplot of Weather')
    # plt.xlabel('Precipitation (PRC)')
    # plt.ylabel('Instances')
    # plt.savefig('Weather target')

    # Pearson correlation
    # plt.figure(figsize=(12, 10))
    # cor = X.corr()
    # pd_cor = pd.DataFrame(cor)
    # pd_cor.to_excel('other_files/'+ f'CORR_weather.xlsx')
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.title('pearson correlation for weather dataset')
    # plt.savefig('images/pearson correlation for weather dataset')

    return X, y


def read_data_forest_cover_type(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "forestCoverType.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, 1:12]
    y = df.iloc[:, -1:].values.flatten()

    # Pearson correlation
    # plt.figure(figsize=(12, 10))
    # cor = X.corr()
    # pd_cor = pd.DataFrame(cor)
    # pd_cor.to_excel('other_files/'+ 'CORR_forestCoverType.xlsx')
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.title('pearson correlation for forestcover dataset')
    # plt.savefig(r'images/pearson correlation for forestcover dataset')

    return X, y


def read_data_anas(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "panama.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1:]

    # Pearson correlation
    # plt.figure(figsize=(12, 10))
    # cor = X.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.title('pearson correlation for anas dataset')
    # plt.savefig('images/pearson correlation for anas dataset')
    # pd_cor = pd.DataFrame(cor)
    # pd_cor.to_excel('other_files/'+ f'CORR_panama.xlsx')

    return X, y


def load_stream(name, drift=True, shuffle=False):
    """
    Available dataset: 'electricity', 'weather', 'forestcover', 'anas'
    Return a stream of the dataset with injected drift if drift is True, and an array of 1 and 0 corresponding
    to the rows with a drift injected
    :param shuffle: Bool, wheter to shuffle or not the dataset
    :param name: string, dataset to load.
    :param drift: Bool, default is True
    :return: skmultiflow datastream, np.array of drifted rows, int of drift starting point
    """

    if name == 'electricity':
        X, y = read_data_electricity_market(shuffle=shuffle)
        if drift:
            X, y, drift_point, drift_cols = inject_drift(X, y)
            drifted_rows = X['drifted']



    elif name == 'weather':
        X, y = read_data_weather(shuffle=shuffle)
        if drift:
            X, y, drift_point, drift_cols = inject_drift(X, y)
            drifted_rows = X['drifted']

    elif name == 'forestcover':
        X, y = read_data_forest_cover_type(shuffle=shuffle)
        if drift:
            X, y, drift_point, drift_cols = inject_drift(X, y)
            drifted_rows = X['drifted']

            stream = DataStream(X.drop(columns=['drifted']), y)

    elif name == 'anas':
        X, y = read_data_anas(shuffle=shuffle)
        if drift:
            X, y, drift_point, drift_cols = inject_drift(X, y, classification=False)
            drifted_rows = X['drifted']

    if drift:
        stream = DataStream(X.drop(columns=['drifted']), y)

    else:
        stream = DataStream(X, y)
        drifted_rows = np.zeros(X.shape[0])
        drift_point = np.nan

    plt.close('all')

    return stream, np.array(drifted_rows), drift_point, drift_cols

