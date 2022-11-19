import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

def LDAOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de LinearDiscriminantAnalysis")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    best_score = 0
    LDA_solver = 'svd'
    solverList = ['svd', 'lsqr', 'eigen']
    # Optimisez la paramétrisation du mlp
    for hp1 in solverList:
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = LinearDiscriminantAnalysis(solver=hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            LDA_solver = hp1
    return LDA_solver

def NearestCentroidOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de NearestCentroid")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    best_score = 0
    NC_metric = 'euclidean'
    distanceMetrics = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean']
    # Optimisez la paramétrisation du mlp
    for hp1 in distanceMetrics:
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = NearestCentroid(metric=hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            NC_metric = hp1
    return NC_metric

def LogisticRegressionOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de LogisticRegression")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    best_score = 0
    LR_solver = 'lbfgs'
    solverList = ['newton-cg', 'lbfgs', 'sag', 'saga']
    # Optimisez la paramétrisation du mlp
    for hp1 in solverList:
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = LogisticRegression(solver = hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            LR_solver = hp1

    best_score = 0
    LR_max_iter = 400
    for hp2 in range(1, 10):
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = LogisticRegression(max_iter=400 * hp2)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            LR_max_iter = 400 * hp2
    return LR_solver, LR_max_iter

def PerceptronOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de Perceptron")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    best_score = 0
    PER_max_iter = 400
    # Optimisez la paramétrisation du mlp
    for hp1 in range(1, 10):
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = Perceptron(max_iter = 400 * hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            Per_max_iter = 400 * hp1

    best_score = 0
    PER_tol = 50
    for hp2 in [float("1e%d" % i) for i in range(-10, 6)]:
        clf = Perceptron(tol =hp2)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            PER_tol = 50 * hp2
    return PER_max_iter, PER_tol

def MLPOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de MLPClassifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

    best_score = 0
    MLP_hidden_layer_sizes = 400
    # Optimisez la paramétrisation du mlp
    for hp1 in range(1, 7):
        # On laisse le paramètre batch_size à sa valeur par défaut
        clf = MLPClassifier(hidden_layer_sizes=400 * hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            MLP_hidden_layer_sizes = 400 * hp1

    best_score = 0
    MLP_batch_size = 50
    for hp2 in range(1, 11):
        clf = MLPClassifier(batch_size=50 * hp2)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            MLP_batch_size = 50 * hp2
    return MLP_hidden_layer_sizes, MLP_batch_size

def KNNOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de KNeighborsClassifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    best_score = 0
    KNN_n_neighbors = 0
    for hp1 in range(1, 11):
        # On laisse la distance p à la valeur par défaut
        clf = KNeighborsClassifier(n_neighbors=hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            KNN_n_neighbors = hp1
    best_score = 0
    KNN_p = 0
    for hp2 in range(1, 6):
        # On fixe le nombre de voisins à la valeure optimale trouvée précedemment
        clf = KNeighborsClassifier(n_neighbors=KNN_n_neighbors, p=hp2)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            KNN_p = hp2
    return KNN_n_neighbors, KNN_p


def SVCOptimalParameters(X,y):
    print("     Calcul des paramètres optimaux de SVC")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    best_score = 0
    SVC_C = 0
    # Optimisez la paramétrisation du SVC
    for hp1 in [float("1e%d" % i) for i in range(-10, 6)]:
        # On laisse le paramètre gamma à sa valeur par défaut
        clf = SVC(C=hp1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            SVC_C = hp1

    # Algorithme pour calculer sigma min.
    distances = np.array([ np.linalg.norm(i - j) for i in X_train for j in X_train])
    nonzero = np.ma.masked_equal(distances,0)
    sigma = np.amin(nonzero)
    best_score = 0
    SVC_gamma = 0
    for hp2 in range(0, 10):
        # On fixe la regularisation au parametre optimal trouvé avant
        clf = SVC(C=SVC_C, gamma = sigma * pow(2, hp2))
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if (score > best_score):
            # On retient le meilleur choix trouvé
            best_score = score
            SVC_gamma = sigma * pow(2, hp2)
    return SVC_C, SVC_gamma