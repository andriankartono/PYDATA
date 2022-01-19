'''
The MNIST-numbers dataset is a set of labeled, single-digit images
which was used as a benchmark for computer vision image
recognition algorithms - search for the data or load it with
tensorflow (ignore the quantum stuff, just go to 1.1). Build a model
for this dataset using SVM, preferably with radial basis functions.
Parameter tuning might be too expensive for your machine (some
hours, probably days on a typical Celeron/2C4T-notebook), so look
up a suitable parameter combination which yields an accuracy of at
least 90% for you (or try LinearSVM, if that's still too slow)!
NOTE: there is numbers-dataset in sklearn, which is different!
Plot a grid of 10 test results, labeled with the true and the
predicted label. (3pts: 1 pts for the setup, 1 pts for right
parameters, 1 pts for a plot)
'''

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_reshaped = np.zeros([len(X_train), 784])

for i in range(len(X_train)):
    X_train_reshaped[i,:] = X_train[i].reshape(1,-1)

X_train_reshaped = X_train_reshaped[0:6000,:]
y_train = y_train[0:6000]

X_test_reshaped = np.zeros([len(X_test), 784])

for i in range(len(X_test)):
    X_test_reshaped[i,:] = X_test[i].reshape(1,-1)

X_test_reshaped = X_test_reshaped[0:1000,:]
y_test = y_test[0:1000]

svc = SVC(kernel='linear', C=0.00001, gamma=1)
svc.fit(X_train_reshaped, y_train)
y_pred = svc.predict(X_test_reshaped)

acc = np.round(accuracy_score(y_pred, y_test), 2)

# Plot
fig, axs = plt.subplots(4, 5, figsize=(32,6))
axs = axs.ravel()
for i in range(20):
    axs[i].imshow(X_test[i], cmap='winter')
    axs[i].set_title(f'Pred: {y_pred[i]}, True: {y_test[i]}')
    axs[i].axis('off')
plt.suptitle(f'accuracy score: {acc}')
plt.tight_layout()
plt.savefig('output_digit.png')
plt.show()