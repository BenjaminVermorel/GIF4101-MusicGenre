import numpy as np
from sklearn import preprocessing
import pandas as pd

classes = ('blues', 'classical', 'country', 'disco',
           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')


def getData(data):
    # remove the non-numeric columns
    numeric_values = data._get_numeric_data()
    y_str = []
    y = []
    # classes values
    for element in data.values:
        y_str.append(element[59])
        i = 0
        while element[59] != classes[i]:
            i += 1
        y.append(i)
    numeric_values = np.array(numeric_values)
    y_str = np.array(y_str)
    y = np.array(y)

    print(numeric_values)

    #on normalise X :
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(numeric_values)

    print(X)

    return X, y, y_str


def extractionDataFromCSV(Path="C:\\Users\\Jules\\Downloads\\Data\\features_30_sec.csv"):
    # comma delimited is the default
    dataSet = pd.read_csv(Path, header=0)

    X, y, y_str = getData(dataSet)

    return X, y, y_str


if (__name__ == "__main__"):
    extractionDataFromCSV()
