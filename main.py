# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from CSVDATA.DuoClasse.DuoClasse import duoClasse
from CSVDATA.MultiClasse.multiClasse import multiClasse

""""
LISTE DES CLASSIFIEURS
- QuadraticDiscriminantAnalysis
- LinearDiscriminantAnalysis
- GaussianNB
- NearestCentroid
- LogisticRegression
- Perceptron
- KNeighborsClassifier
- MLPClassifier
- SVC
- deep-learning/convolution
"""

#Getting images from GTZAN
rootPath = "C:\\archive\\Data"
#Variable déterminant si l'on doit calculer les valeurs optimales à utiliser pour le classifieur
#Sinon, utilise des valeurs par défauts. Cette option est très chronophage quand mise à False.
usePreCalculatedParams = False

#Differents programmes du projet
""""
duoClasse(rootPath, usePreCalculatedParams)
"""
multiClasse(rootPath, usePreCalculatedParams)

