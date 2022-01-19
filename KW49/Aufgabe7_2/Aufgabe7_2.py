'''
Get the data from wave.csv on Moodle (which represents a very
noisy measurement of the function f = exp(-(x/4)^2) · cos(4 · x)).
Use GridSearchCV to find an optimized KRR-model with a
“rbf”-kernel-function (change the Kernels alpha and gamma in
hyperparameter-tuning). Use a training set of 80%, 5-fold
cross-validation and the MSE-scoring metric. Make a plot showing
the “true” function f , the datapoints used for training, (in another
color/markerstyle) the datapoints used for testing and the
evaluated model f. Indicate the MSE, MAE and R2-score on the
test-data in the plot title.
Make a short argument how you think, the model could be
improved, provided you could evaluate arbitrary points as input for'''

import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

def f(x):
    return np.exp(-(x/4)**2) * np.cos(4*x)

df=pd.read_csv("wave.csv")

x=df["x"]
y=df["y"]

x_train, x_test, y_train, y_test= train_test_split(x,y,shuffle=True, train_size=0.8)
krr=KernelRidge(kernel='rbf')
krr_param= {'alpha': np.arange(0.01, 0.1 , 0.01), 'gamma': np.arange(2,4.5,0.1)}
hyperparam=GridSearchCV(krr, krr_param, scoring='neg_mean_squared_error', cv=5)

hyperparam.fit(x_train.values.reshape(-1,1), y_train)
y_pred=hyperparam.predict(x_test.values.reshape(-1,1))

alpha=hyperparam.best_params_.get('alpha')
gamma=hyperparam.best_params_.get('gamma')

mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)

x_plot=np.linspace(-10,10,10000)

y_plot= hyperparam.predict(x_plot.reshape(-1,1))

plt.plot(x_plot, f(x_plot), color='red')
plt.plot(x_plot, y_plot)
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.yticks([-1,-0.5, 0, 0.5, 1.0])
plt.legend(['true_f', 'predicted f', 'training data', 'test data'])
plt.title('MSE: {:.3f}, MAE: {:.3f}, $R^2$: {:.3f}'.format(mse, mae, r2))
plt.savefig("wave.png")

#When comparing the true function and the function of the given dataset, we can see that there are differences in the y value.
#This will cause the predicted function to have errors compared to the true function.
#One of the ideas that we can use is to value some points more than other points -> better points are more important in training and noises need to be filtered as much as possible.
#This should reduce the effect of 'noises' on the predicted function.
#Another improvement is to increase number of kfoldvalidation and use gridsearchcv for other parameters as well.

#Limitations:
#Limited number of data for training and testing -> more data = better prediction
#Existence of 'Noise' which will always cause the function to not be 100% correct.