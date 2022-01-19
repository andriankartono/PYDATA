'''
Use linear regression to find the parameters a and b to fit the
function y = asin(x) + bx to the data in sin .csv. Plot the
regression model and the datapoints. Hand in code and the plot
(with an indication of your fitted parameters) (2pts: 1 pts for
implementing the algorithm with “simple” matrix ops from
numpy/scipy, 0.5 pts for the results, 0.5 pts for a suitable plot)
Notice: f (x) = sin(x) can be easily integrated into the OLS
framework by using it as an “extra” feature z = sin(x).
'''

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import pandas as pd
import math

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

#dataset preparation
sin_dataset=pd.read_csv("sin.csv")
x=sin_dataset["x"]
expected_y=sin_dataset["y"] #result
sin_x=np.sin(x)#input: y=a*sin_x + bx
plt.scatter(x,expected_y)

x_concat=pd.concat([sin_x, x], axis=1)
x_concat.columns=["sin_x", "x"]

#prepare the x and y matrix in required forms
y_numpy=expected_y.to_numpy()
y_matrix=np.asmatrix(y_numpy)
y_matrix=y_matrix.transpose()

x_numpy=x_concat.to_numpy()
x_matrix=np.asmatrix(x_numpy)
x_matrix_transpose=x_matrix.transpose()

#implementation in the formula
xTx_matrix=np.matmul(x_matrix_transpose, x_matrix)
xTx_matrix_inverse=inv(xTx_matrix)
temp_matrix=np.matmul(xTx_matrix_inverse, x_matrix_transpose)
final_matrix=np.matmul(temp_matrix, y_matrix)

x_line=np.linspace(-10,10,1000)
y_line=2.8612775*np.sin(x_line)-3.15279497*x_line
plt.suptitle("OSL Fit, a=2.8612775 , b=-3.15279497")
plt.plot(x_line, y_line , label='OLS')
plt.legend(loc='upper right')
plt.savefig("OSL.png")
plt.show()

#Notes:
#Reference other than script: https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
#y=a*sin_x + bx can be transformed into y=a*x1+ b*x2
#transform to a matrix notation -> (y1 y2 ...)T = (X1 X2)(a b)T
#dim(Y)=N*1, dim(X)=N*2, dim(constants)=2*1
#constant= (xT*x)^-1 * xT * y