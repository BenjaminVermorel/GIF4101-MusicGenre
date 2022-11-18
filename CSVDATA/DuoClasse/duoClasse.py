import os
import pandas as pd

def duoClasse(rootPath):
    featuresFilePath = os.path.join(rootPath, "features_30_sec.csv")

    # comma delimited is the default
    dataSet = pd.read_csv(featuresFilePath, header=0)
    original_headers = list(dataSet.columns.values)
    X, y = getDataByClass(dataSet, 'blues')
    print(y)
    # create a numpy array with the numeric values for input into scikit-learn
    numeric_headers = list(X.columns.values)
    numpy_array = X.values






def getDataByClass(data, classe):
    # remove the non-numeric columns
    numeric_values = data._get_numeric_data()

    y = []
    # classes values
    for element in data.values:
        if element[59] == classe:
            y.append(1)
        else:
            y.append(0)
    return numeric_values, y