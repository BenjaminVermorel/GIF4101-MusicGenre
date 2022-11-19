import numpy as np
from sklearn.model_selection import train_test_split


def clfGroupScore(clf, X, y, groupValues = False):
    if groupValues:
        # Prend 1 si la classe du groupe est bonne, 0 sinon
        paquetsCorrects = 0
        prediction = clf.predict(X)
        for groupIndex in range(int(len(X)/10)):
            valeurCorrect = 0
            # On regarde combien d'elements dans le groupe ont la bonne prediction
            for i in range(10):
                if prediction[groupIndex * 10 + i] == y[groupIndex * 10 + i]:
                    valeurCorrect = valeurCorrect + 1
            # Si la majorite des predictions dans le groupe est bonne
            # on considere que la prediction de groupe est bonne
            if valeurCorrect > 5:
                paquetsCorrects = paquetsCorrects + 1
        score = paquetsCorrects/int(len(X)/10)
        scores = np.array(score)
        return scores.mean()
    else:
        return clf.score(X, y)

def getTrainTestIndexes(taille, pourcentage):
    indexPaquets = []
    for i in range(taille):
        indexPaquets.append(i)
    indexPaquets = np.array(indexPaquets)
    np.random.shuffle(indexPaquets)
    trainIndex = []
    testIndex = []
    for i in range(int(taille*pourcentage)):
        trainIndex.append(indexPaquets[i])
    for i in range(int(taille*pourcentage), taille ):
        testIndex.append(i)
    return trainIndex, testIndex

def getValueByIndexPaquets(X, y, indexes):
    X_values = []
    y_values = []
    for index in indexes:
        for i in range(10):
            X_values.append(X[index * 10 + i])
            y_values.append(y[index * 10 + i])
    return X_values, y_values

def train_test_customSplit(X,y, groupValues = False):
    if groupValues:
        trainIndex, testIndex = getTrainTestIndexes(999, 0.75)
        X_train, y_train = getValueByIndexPaquets(X, y, trainIndex)
        X_test, y_test = getValueByIndexPaquets(X, y, testIndex)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    return X_train, X_test, y_train, y_test