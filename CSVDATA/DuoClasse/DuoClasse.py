import os
import collections
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize.linesearch import LineSearchWarning

from CSVDATA.DuoClasse.OptimParamClassifieurs import *

classes = ('blues', 'classical', 'country', 'disco',
           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=LineSearchWarning)
@ignore_warnings(category=UserWarning)
def duoClasse(rootPath, usePrecalculatedParam=True):
    trenteSecondePath = os.path.join(rootPath, "features_30_sec.csv")
    dfTrenteSecondes = testScore(trenteSecondePath, usePrecalculatedParam)

    troisSecondePath = os.path.join(rootPath, "features_3_sec.csv")
    dfTroisSecondes = testScore(troisSecondePath, usePrecalculatedParam)

    dfTroisSecondesGroupe = testScore(troisSecondePath, usePrecalculatedParam, groupValues=True)

    print(dfTrenteSecondes.to_string())
    print(dfTroisSecondes.to_string())
    print(dfTroisSecondesGroupe.to_string())


def testScore(Path, usePrecalculatedParam=True, groupValues=False):
    # comma delimited is the default
    dataSet = pd.read_csv(Path, header=0)
    original_headers = list(dataSet.columns.values)
    numeric_values = dataSet._get_numeric_data()
    numeric_headers = list(numeric_values.columns.values)

    df = pd.DataFrame()
    for element in classes:
        print("Travail sur la classe " + element)
        X, y = getDataByClass(dataSet, element)
        scores = train(X, y, usePrecalculatedParam)
        df[element] = scores
        # df = pd.DataFrame(scores, index=[element])

    df['Moyenne'] = df.mean(axis=1)
    df = df.sort_values(by='Moyenne', axis='index', ascending=False)
    return df


def train(X, y, usePrecalculatedParam, groupValues=False):
    classifiers = calculateParameters(X, y, usePrecalculatedParam)
    # Dictionnaire pour enregistrer les erreurs selon les classifieurs
    scores = collections.OrderedDict()
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        # Validation croisée (K=3) à faire
        rkf = KFold(n_splits=3, shuffle=True)
        errors = []
        if (groupValues):
            # On gérer les données comme etant des paquets
            # On ne peut donc pas faire de split() directement sur X
            indexPaquets = []
            for i in range(1000):
                indexPaquets.append(i)

            for train_index, test_index in rkf.split(indexPaquets):
                X_train, y_train = getValueByIndexPaquets(X,y, train_index)
                X_test, y_test = getValueByIndexPaquets(X,y, test_index)
                # On entraine le pli et on calcule son erreur
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                errorCount = sum(i != j for i, j in zip(pred, y_test))
                err_test = errorCount / len(X_test)
                errors.append(err_test)
        else:
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

def getValueByIndexPaquets(X, y, indexes):
    X_train, X_test = [], []
    y_train, y_test = [], []
    for index in indexes:
        for i in range(10):
            X_train.append(X[index * 10 + i])
            y_train.append(y[index * 10 + i])
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


def calculateParameters(X, y, usePreCalculatedParam):
    if (usePreCalculatedParam):
        # Ces paramètres ne sont qu'une valeur par défaut permettant de gagner
        # sur l'execution du programme. Les résultats ne seront pas optimals.
        LDA_solver = 'svd'
        NC_metric = 'manhattan'
        LR_solver, LR_max_iter = 'newton-cg', 800
        PER_max_iter, PER_tol = 400, 5e-09
        KNN_n_neighbors, KNN_p = 5, 2
        MLP_hidden_layer_sizes, MLP_batch_size = 800, 200
        SVC_C, SVC_gamma = 1e-10, 3200
    else:
        LDA_solver = LDAOptimalParameters(X, y)
        NC_metric = NearestCentroidOptimalParameters(X, y)
        LR_solver, LR_max_iter = LogisticRegressionOptimalParameters(X, y)
        PER_max_iter, PER_tol = PerceptronOptimalParameters(X, y)
        KNN_n_neighbors, KNN_p = KNNOptimalParameters(X, y)
        MLP_hidden_layer_sizes, MLP_batch_size = MLPOptimalParameters(X, y)
        SVC_C, SVC_gamma = SVCOptimalParameters(X, y)

    classifiers = [
        QuadraticDiscriminantAnalysis(tol=0),
        LinearDiscriminantAnalysis(solver=LDA_solver),
        GaussianNB(),
        NearestCentroid(metric=NC_metric),
        LogisticRegression(solver=LR_solver, max_iter=LR_max_iter),
        Perceptron(max_iter=PER_max_iter, tol=PER_tol),
        KNeighborsClassifier(n_neighbors=KNN_n_neighbors, p=KNN_p),
        MLPClassifier(hidden_layer_sizes=MLP_hidden_layer_sizes, batch_size=MLP_batch_size),
        SVC(C=SVC_C, gamma=SVC_gamma),
    ]
    return classifiers
