import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn import linear_model, neighbors, preprocessing
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

iris=datasets.load_iris(as_frame=True)
y=iris.target
x=iris.data
x=x.drop(columns=["petal length (cm)","petal width (cm)"])

h=0.005
# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

fig,axs=plt.subplots(2,2)
axs=axs.ravel()

x_train,x_test, y_train, y_test = train_test_split(x.values, y.values, shuffle=True, test_size=0.3)

#KNN
neighbor_param = list(range(1,50))
kscore=[]
knnparam = dict(n_neighbors=neighbor_param)
knn=KNeighborsClassifier()

#SVMLinear
svmlinear= SVC(kernel="linear")
svmlinear_param={'C': [0.01,0.1,1,10,100,1000]}

#SVMPOLY
svmpoly = SVC(kernel = "poly")
svmpoly_param={'C': [0.01, 0.1, 1, 5, 10, 50, 100], 'coef0': [0, 0.5], 'degree': np.arange(2,5,1)}

#SVMRBF
svmrbf = SVC(kernel='rbf')
svmrbf_param = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

for i in range(4):
    x_min, x_max = x["sepal length (cm)"].min() -1 , x["sepal length (cm)"].max() +1
    y_min, y_max = x["sepal width (cm)"].min() - 1 , x["sepal width (cm)"].max() +1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max, h))

    r1,r2 = xx.flatten(), yy.flatten()
    r1,r2= r1.reshape((len(r1),1)), r2.reshape((len(r2),1))
    color_plot_stack= np.hstack((r1,r2))

    if i == 0:
        modeltitle = "KNN"
        modeltype = knn
        modelparam = knnparam
    elif i == 1:
        model = "linear"
        modeltitle = "linear SVC"
        modeltype = svmlinear
        modelparam = svmlinear_param
    elif i == 2:
        model = "poly"
        modeltitle = "poly SVC"
        modeltype = svmpoly
        modelparam = svmpoly_param
    elif i == 3:
        model = "rbf"
        modeltitle = "rbf SVC"
        modeltype = svmrbf
        modelparam = svmrbf_param

    hyperparam=GridSearchCV(modeltype, modelparam, cv=5, n_jobs=-1, scoring="accuracy")
    hyperparam.fit(x_train,y_train)
    bestparam=hyperparam.best_params_

    y_train_pred = hyperparam.predict(x_train)
    y_test_pred = hyperparam.predict(x_test)

    acc_train=np.round(accuracy_score(y_train,y_train_pred),2)
    acc_test=np.round(accuracy_score(y_test,y_test_pred),2)

    Z = hyperparam.predict(color_plot_stack)
    Z = Z.reshape(xx.shape)
    axs[i].pcolormesh(xx,yy,Z, cmap=cmap_light, shading='auto')
    #print(i)

    sns.scatterplot(x=x["sepal length (cm)"], y=x["sepal width (cm)"], hue=iris.target_names[y], palette=cmap_bold, alpha=1.0, edgecolor="black", ax=axs[i])
    axs[i].set_xlim(xx.min(), xx.max())
    axs[i].set_ylim(yy.min(), yy.max())
    axs[i].legend(loc="upper right")
    if i == 0:
        axs[i].set_title(f"{modeltitle}, train: {acc_train},test: {acc_test} \n {bestparam}")
    else:
        axs[i].set_title(f"{modeltitle}, train: {acc_train},test: {acc_test} \n {bestparam}, 'kernel': '{model}'")
    axs[i].set(xlabel=None)
    axs[i].set(ylabel=None)

# Get current figure size
current_figsize = fig.get_size_inches()
# Set the figure size as the double of the original one
# Otherwise the saved figure is shrinked
fig.set_size_inches(current_figsize * 2)
plt.savefig("Aufgabe8_1.png")
plt.show()
        
