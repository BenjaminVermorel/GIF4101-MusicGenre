
def LDAScore(clf, X,y, groupValues = False):
    if groupValues:
        prediction = clf.predict(X)
    else:
        return clf.score(X,y)