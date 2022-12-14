import numpy as np
from sklearn.model_selection import train_test_split


def clfScore(clf, X, y, groupValues = False):
    if groupValues:
        # Prend 1 si la classe du groupe est bonne, 0 sinon
        paquetsCorrects = 0
        prediction = clf.predict(X)
        for groupIndex in range(int(len(X)/10)):
            valeurCorrect = [0,0,0,0,0,0,0,0,0,0]
            # On regarde combien d'elements dans le groupe ont la bonne prediction
            for i in range(10):
                valeurCorrect[prediction[groupIndex * 10 + i]] = valeurCorrect[prediction[groupIndex * 10 + i]] + 1
            classeCorrecte = y[groupIndex * 10]
            egalite = False
            classeIndex = -1
            max = -1
            for i in range(10):
                if(valeurCorrect[i] > max):
                    max = valeurCorrect[i]
                    classeIndex = i
                    egalite = False
                if(valeurCorrect[i] == max):
                    egalite = True
            #Le groupe est considere comme correct si on a une majorité dans la bonne classe
            #En cas d'egalite, la prediction n'est pas correcte
            if(egalite == False):
                if(classeIndex == classeCorrecte):
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