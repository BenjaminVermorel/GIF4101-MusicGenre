import os
import collections
import pandas as pd
import numpy as np
from IPython import display
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
classes = ('blues', 'classical', 'country', 'disco',
           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')

classifiers = [
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis(),
        GaussianNB(),
        NearestCentroid(),
        LogisticRegression(solver = 'lbfgs', tol =1e-1, max_iter =1000),
        Perceptron(tol = 1e-2, alpha = 1e-05, max_iter =10000),
        KNeighborsClassifier(),
        MLPClassifier(),
        SVC(),
    ]

def duoClasse(rootPath):
    featuresFilePath = os.path.join(rootPath, "features_30_sec.csv")

    # comma delimited is the default
    dataSet = pd.read_csv(featuresFilePath, header=0)
    original_headers = list(dataSet.columns.values)
    numeric_values = dataSet._get_numeric_data()
    numeric_headers = list(numeric_values.columns.values)

    df = pd.DataFrame()
    for element in classes:
        X, y = getDataByClass(dataSet, element)
        scores = train(X, y)
        df[element] = scores
        #df = pd.DataFrame(scores, index=[element])

    df['Moyenne'] = df.mean(axis=1)
    df = df.sort_values(by = 'Moyenne', axis = 'index', ascending = False)
    print(df.to_string())


def train(X, y):
    # Dictionnaire pour enregistrer les erreurs selon les classifieurs
    scores = collections.OrderedDict()
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        # Validation croisée (K=3) à faire
        rkf = KFold(n_splits=3, shuffle=True)
        errors = []
        for train_index, test_index in rkf.split(X):
            # On genere les indexes d'un plis
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # On entraine le pli et on calcule son erreur
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            errorCount = sum(i != j for i, j in zip(pred, y_test))
            err_test = errorCount / len(X_test)
            errors.append(err_test)
        validError = np.mean(errors)
        scores[clf_name] = 1 - validError
    return scores

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

    numeric_values = np.array(numeric_values)
    y = np.array(y)
    return numeric_values, y