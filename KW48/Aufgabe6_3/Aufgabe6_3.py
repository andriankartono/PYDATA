'''
K-nearest neighbors classification: Get the iris .data-dataset,
either like shown in the pandas-intro or from
sklearn.datasets. load iris . Then make a scatterplot of the first
two columns (“sll/cm”, “sp/cm”) so that each “type” (last
column) has an individual color or marker. Now apply a
KNN-classifier (from sklearn.neighbors.KNeighborsClassifier and
fit the plotted columns to the category. Then use axes.pcolormesh
to plot the evaluation of the classifier model on a grid. Repeat for
1, 5, 10, 50 “k-neighbors” and plot in a single plot as shown at the
end.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import matplotlib.gridspec as gridspec

n_neighbors = [1,5,10,50]

# import some data to play with
iris = datasets.load_iris()

fig2 = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)


# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = 0.02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

weights="uniform"
counter=0

for n_neighbor in n_neighbors:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbor, weights=weights)
    clf.fit(X, y)

    if(counter==0):
        ax = fig2.add_subplot(spec2[0, 0])
    elif(counter==1):
        ax= fig2.add_subplot(spec2[0, 1])
    elif(counter==2):
        ax= fig2.add_subplot(spec2[1, 0])
    elif(counter==3):
        ax= fig2.add_subplot(spec2[1, 1])
    #print(counter)
    counter+=1

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.figure(figsize=(8, 6))
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # Plot also the training points
    sns.scatterplot(x=X[:, 0],y=X[:, 1],hue=iris.target_names[y],palette=cmap_bold,alpha=1.0,edgecolor="black",)
    #plt.pcolormesh([X[:,0], X[:,1]] , cmap='rainbow')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbor, weights), fontsize=7
    )
    ax.set_xlabel(iris.feature_names[0], fontsize=5)
    ax.set_ylabel(iris.feature_names[1], fontsize=5)

plt.savefig("Aufgabe6_3.png")
plt.show()