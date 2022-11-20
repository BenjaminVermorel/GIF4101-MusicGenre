import os
import collections
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize.linesearch import LineSearchWarning

from CSVDATA.DuoClasse.DuoClasse import calculateParameters
from CSVDATA.DuoClasse.ScoreClassifieurs import *

@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=LineSearchWarning)
@ignore_warnings(category=UserWarning)
def multiClasse(rootPath, usePrecalculatedParam=True):

    trenteSecondePath = os.path.join(rootPath, "features_30_sec.csv")
    dfTrenteSecondes = testScore(trenteSecondePath, usePrecalculatedParam)

    troisSecondePath = os.path.join(rootPath, "features_3_sec.csv")
    #dfTroisSecondes = testScore(troisSecondePath, usePrecalculatedParam)
    dfTroisSecondesGroupe = testScore(troisSecondePath, usePrecalculatedParam, groupValues=True)

    #print(dfTrenteSecondes.to_string())
    #print(dfTroisSecondes.to_string())
    print(dfTroisSecondesGroupe.to_string())

def testScore(Path, usePrecalculatedParam=True, groupValues=False):
    # comma delimited is the default
    dataSet = pd.read_csv(Path, header=0)
    original_headers = list(dataSet.columns.values)
    numeric_values = dataSet._get_numeric_data()
    numeric_headers = list(numeric_values.columns.values)

    df = pd.DataFrame()
    X, y = getDataMultiClass(dataSet)
    scores = train(X, y, usePrecalculatedParam, groupValues)
    df['score'] = scores

    df = df.sort_values(by='score', axis='index', ascending=False)
    return df

def getDataMultiClass(data):
    # remove the non-numeric columns
    numeric_values = data._get_numeric_data()
    y = []
    # classes values
    for element in data.values:
        y.append(element[59])
    numeric_values = np.array(numeric_values)
    y = np.array(y)
    return numeric_values, y

def train(X, y, usePrecalculatedParam, groupValues=False):
    classifiers = calculateParameters(X, y, usePrecalculatedParam, groupValues)
    # Dictionnaire pour enregistrer les erreurs selon les classifieurs
    clfScores = collections.OrderedDict()
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        # Validation croisée (K=3) à faire
        rkf = KFold(n_splits=3, shuffle=True)
        scores = []
        if (groupValues):
            # On gérer les données comme etant des paquets
            # On ne peut donc pas faire de split() directement sur X
            indexPaquets = []
            for i in range(999):
                indexPaquets.append(i)

            for train_index, test_index in rkf.split(indexPaquets):
                X_train, y_train = getValueByIndexPaquets(X,y, train_index)
                X_test, y_test = getValueByIndexPaquets(X,y, test_index)
                # On entraine le pli et on calcule son erreur
                clf.fit(X_train, y_train)
                score = clfScore(clf, X_test, y_test, True)
                scores.append(score)
        else:
            for train_index, test_index in rkf.split(X):
                # On genere les indexes d'un plis
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # On entraine le pli et on calcule son erreur
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                scores.append(score)
        validScore = np.mean(scores)
        clfScores[clf_name] = validScore
    return clfScores