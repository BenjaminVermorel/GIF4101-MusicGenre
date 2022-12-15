import numpy
from matplotlib import pyplot, offsetbox
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import plotly.express as px
from plotly.offline import plot

from CSVDATA.ExtractionDataFromCSV.ExtractionDataFromCSV import extractionDataFromCSV


# =======================================================================
#               2D
# =======================================================================
def plot_clustering_2D(X_red, y, labels, title, savepath):
    # Tiré de https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html
    # Auteur : Gael Varoquaux
    # Distribué sous license BSD
    #
    # - X_red: array numpy contenant les caractéristiques (features)
    #   des données d'entrée, réduit à 2 dimensions / numpy array containing
    #   the features of the input data of the input data, reduced to 2 dimensions
    #
    # - labels: un array numpy contenant les étiquettes de chacun des
    #   éléments de X_red, dans le même ordre. / a numpy array containing the
    #   labels of each of the elements of X_red, in the same order.
    #
    # - title: le titre que vous souhaitez donner à la figure / the title you want
    #   to give to the figure
    #
    # - savepath: le nom du fichier où la figure doit être sauvegardée / the name
    #   of the file where the figure should be saved
    #
    x_min, x_max = numpy.min(X_red, axis=0), numpy.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    pyplot.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        pyplot.text(X_red[i, 0], X_red[i, 1], str(labels[i]),
                    color=pyplot.cm.nipy_spectral(y[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(title, size=17)
    pyplot.axis('off')
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.show()


def pca_2d():
    X, y, y_str = extractionDataFromCSV()

    pca = PCA(n_components=2).fit(X)
    X_PCA = pca.transform(X)
    plot_clustering_2D(X_PCA, y, y_str, "Réduction de X en utilisant PCA", "")


def tsne_2d():
    X, y, y_str = extractionDataFromCSV()

    X_tsne = TSNE(n_components=2).fit_transform(X)
    plot_clustering_2D(X_tsne, y, y_str, "Réduction de X en utilisant tSNE", "")


def mds_2d():
    X, y, y_str = extractionDataFromCSV()

    X_MDS = MDS(n_components=2, n_init=1).fit_transform(X)
    plot_clustering_2D(X_MDS, y, y_str, "Réduction de X en utilisant MDS", "")


# =======================================================================
#               3D
# =======================================================================
def pca_3d():
    X, y, y_str = extractionDataFromCSV()

    pca = PCA(n_components=3).fit(X)
    X_PCA = pca.transform(X)
    fig = px.scatter_3d(X_PCA, x=0, y=1, z=2, color=y_str)
    fig.update_traces(marker_size=2)
    plot(fig)

def tsne_3d():
    X, y, y_str = extractionDataFromCSV()

    X_tsne = TSNE(n_components=3).fit_transform(X)
    fig = px.scatter_3d(X_tsne, x=0, y=1, z=2, color=y_str)
    fig.update_traces(marker_size=2)
    plot(fig)

def mds_3d():
    X, y, y_str = extractionDataFromCSV()

    X_MDS = MDS(n_components=3, n_init=1).fit_transform(X)
    fig = px.scatter_3d(X_MDS, x=0, y=1, z=2, color=y_str)
    fig.update_traces(marker_size=2)
    plot(fig)

if __name__ == "__main__":
    #pca_3d()
    #tsne_3d()
    mds_3d() # /!\ plutot long