'''
2. Now, for the data in sba small.csv on Moodle (taken from here),
build a logistic model which can predict whether for a given
business-case, the loan is predicted to be paid off in time (that’s
the column Defaultfor past data).
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

'''
2.1 read in the dataset and plot a histogramm of the Default and the
CreateJob-column. (0.5 pts)
'''
#Load Dataframe
df=pd.read_csv("sba_small.csv")
fig, axs= plt.subplots(2,1)
x1= df["CreateJob"]
x2= df["Default"]

#Plot CreateJob (max = 130)
axs[0].hist(x1, bins=np.linspace(0,130,131)-0.5)
axs[0].set_title("CreateJob")

#Plot 
axs[1].hist(x2, bins=np.linspace(0,2,3))
axs[1].set_title("Default")
axs[1].set_xticks(np.linspace(0,2,3))

current_figsize=fig.get_size_inches()
fig.set_size_inches(current_figsize*1.5)
plt.savefig("2_1.png")

'''
2.2 now split into the (exercise-setup-recommended) train/test-sets by
the Selected-column, save both files as csvs. (0.5 pts)
'''

df_zero=df[df["Selected"]==0]
df_one=df[df["Selected"]==1]

df_zero.to_csv("Selected0.csv", index=False)
df_one.to_csv("Selected1.csv", index=False)

'''
2.3 finally build a logistic model for the prediction, using the columns
Recession, RealEstate and Portion as inputs.
Calculate and print out (or make a color-coded plot, as you can see
sometimes on the internet), the confusion matrix (with:
sklearn.metrics.confusion matrix). Plot the ROC using the sklearn
provided function sklearn.metrics.roc curve (or do it yourself).
Check out the documentation (or visit the tutorial ;)) to find out
how to get the y score-array.
(3 pts, 1 pts for doing the classification, 1 pts for the confusion
matrix, 2 pts for plotting the ROC-curve - put in the code into the
last exercise please on hand-in!)
'''

x = df[['Recession', 'RealEstate', 'Portion']]
y = df['Default']

x_train= df_one[["Recession", "RealEstate", "Portion"]]
y_train= df_one["Default"]

x_test= df_zero[["Recession", "RealEstate", "Portion"]]
y_test= df_zero["Default"]

logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred= logreg.predict(x_test)

confusion_matrix= metrics.confusion_matrix(y_test, y_pred)

cmd= metrics.ConfusionMatrixDisplay(confusion_matrix)
cmd.plot()
plt.savefig("confusionmatrix.png")

fig,ax=plt.subplots()
y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="ROC, auc="+str(auc), color='orange')
plt.plot([0,1],[0,1],'--', color='blue')
plt.legend(loc=4)
plt.savefig('ROCcurve.pdf')

plt.show()