'''
Get the credit data-set from the ISL-book. Now, build a LASSO
machine learning model for the balance depending on
rating, limit , cards, income. Evaluate the LASSO method for
the alpha-regularization parameter ranging from 0 to 10000. Set
aside a test-set of 20% and for each alpha calculate a test-result
with the “trained” model. Make a plot showing the model
coefficients for each alpha in one part and the R2-score on the test
set in another one (with a shared axis).
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv("credit.csv")
y=df["Balance"]
X=df.iloc[:,1:5]
string_list=["Income","Limit", "Rating", "Cards"]
x_train, x_test, y_train, y_test= train_test_split(X, y, shuffle=True, train_size=0.2)
lasso_alphas=np.logspace(0,4,100)
Coef_List=[]
Metric_List=[]

for alphas in lasso_alphas:
    lasso=Lasso(alpha=alphas)
    lasso.fit(x_train,y_train)
    y_pred=lasso.predict(x_test)
    Metric_List.append(r2_score(y_pred,y_test))
    Coef_List.append(lasso.coef_)
    
Array=np.array(Coef_List)

fig= plt.figure()
spec=fig.add_gridspec(ncols=1, nrows=2)

ax0=fig.add_subplot(spec[0,0])
ax0.set_xscale('log')
ax0.set_xlim(right=10500)
ax0.set_ylim(-9,9)
ax0.set_ylabel("Value of the coefficient")
ax0.plot(lasso_alphas, Array[:,0], label="Income")
ax0.plot(lasso_alphas, Array[:,1], label="Limit")
ax0.plot(lasso_alphas, Array[:,2], label="Ratings")
ax0.plot(lasso_alphas, Array[:,3], label="Cards")
ax0.legend(loc='lower left')

ax1=fig.add_subplot(spec[1,0], sharex=ax0)
ax1.plot(lasso_alphas, Metric_List)
ax1.set_ylabel("$R^2$")
ax1.set_xlabel("alpha")

plt.savefig("Aufgabe7_1.png")