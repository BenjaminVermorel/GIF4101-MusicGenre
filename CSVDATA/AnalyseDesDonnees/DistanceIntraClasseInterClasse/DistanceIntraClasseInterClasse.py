import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from CSVDATA.AnalyseDesDonnees.ExtractionDataFromCSV.ExtractionDataFromCSV import extractionDataFromCSV

X, y, y_str = extractionDataFromCSV()

classes = ('blues', 'classical', 'country', 'disco',
           'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')

# Distances intraclasse et interclasse
intraclasse = []
interclasse_glob = []
for i in range(len(classes)):
    intraclasse.append(np.mean(cdist(X[y == i], X[y == i])))
    interclasse_glob.append(np.mean(cdist(X[y == i], X[y != i])))

ratio_intra_inter = np.divide(intraclasse, interclasse_glob)

plt.figure(figsize=(8, 6))
plt.hist(ratio_intra_inter, ec='violet', fc='blue', range=(0, 1), bins=20)
plt.title('Ratios intraclasse-interclasse')
plt.show()
