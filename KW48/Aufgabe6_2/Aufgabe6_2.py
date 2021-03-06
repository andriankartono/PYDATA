'''
2. The Levenberg-Marquardt algorithm: Implement the LMA for a 1
dimensional dependent variable and 1 dimensional feature. Fit the
data in breit wigner.csv to a Breit-Wigner function parametrized
as follows:
g(x) = a/((b - x)**2 + c)
2 + c
For our cases ignore the error-weighting (c = 1) and use central
differences to calculate the Jacobian for g(x) (for example with
stepsize h = 0.001). Stop iterating at a relative change of
β <~ 0.1%. On moodle there is additional information on choosing
the λ-parameter.
Plot the datapoints and the fitted model. Specify the values for a,
b, c in the plot!
'''
from numpy import random
from numpy.typing import _128Bit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
import random

df=pd.read_csv("breit_wigner.csv")

#OK:calculate the result of the breit wigner function
def breit_wigner(a,b,c,x):
    result= a/((b-x)**2+c)
    return result

#OK: calculate the dericative(central derivative method) of the breit wigner function to use for finding the jacobi matrix
def derivative(a,b,c,x, d_param):
    step=0.001
    if(d_param=="a"):
        temp_plus=a+step
        temp_minus=a-step
        return (breit_wigner(temp_plus,b,c,x)-breit_wigner(temp_minus, b, c, x))/(2*step)
    elif(d_param=="b"):
        temp_plus=b+step
        temp_minus=b-step
        return (breit_wigner(a,temp_plus,c,x)-breit_wigner(a, temp_minus, c, x))/(2*step)
    elif(d_param=="c"):
        temp_plus=c+step
        temp_minus=c-step
        return (breit_wigner(a,b,temp_plus,x)-breit_wigner(a, b, temp_minus, x))/(2*step)

#OK: calculate the jacobi matrix making use of the derivative function
def jacobi_matrix(a,b,c, array):
    list_a=[]
    list_b=[]
    list_c=[]

    for x in array:
        list_a.append(derivative(a,b,c,x, "a"))
        list_b.append(derivative(a,b,c,x, "b"))
        list_c.append(derivative(a,b,c,x, "c"))
    
    Series_a, Series_b, Series_c= np.array(list_a), np.array(list_b), np.array(list_c)
    Jacobi=np.column_stack((Series_a,Series_b, Series_c))
    return Jacobi

#OK: calculate error= y-g(current parameters)
def error(a,b,c,expected_result, indep_var):
    return expected_result-breit_wigner(a,b,c, indep_var)

#TBC:calculate the change in the constants
def calculate_delta_const(a,b,c, lamda):
    jacobi=jacobi_matrix(a,b,c, df["x"])
    jacobi_transpose=np.transpose(jacobi)
    product= np.matmul(jacobi_transpose, jacobi)
    sum_product=product + lamda*np.identity(3)
    sum_product_inv=inv(sum_product)
    delta_const=np.matmul(np.matmul(sum_product_inv, jacobi_transpose),error(a,b,c, df["g"], df["x"]))
    return delta_const

#TBC:
def calculate_lamda_0(a,b,c):
    jacobi=jacobi_matrix(a,b,c,df["x"])
    jacobi_transpose=np.transpose(jacobi)
    product=np.matmul(jacobi_transpose,jacobi)
    product_norm=norm(product)
    return product_norm

#TBC: calculate the change in parameter. if the result is >0 -> the error is reduced. if result <0-> the error increased
def calculate_param_change(a_old, b_old, c_old, a_new, b_new, c_new , lamda, delta_constant):
    #jacobi shape-> (9,3)
    jacobi=jacobi_matrix(a_old, b_old, c_old, df["x"])
    #jacobi=jacobi_matrix(a_new, b_new, c_new, df["x"])
    jacobi_transpose=np.transpose(jacobi)
    e0=error(a_old,b_old,c_old, df["g"], df["x"])
    e1=error(a_new,b_new, c_new, df["g"], df["x"])
    nominator= np.dot(e0,e0)-np.dot(e1,e1)
    delta_constant=np.reshape(delta_constant, (3,1))
    temp1= np.transpose(delta_constant)
    temp2= lamda*delta_constant
    temp3=np.matmul(jacobi_transpose,error(a_old, b_old, c_old, df["g"], df["x"]))
    temp3=np.reshape(temp3, (3,1))
    denominator= np.dot(temp1, (temp2+ temp3))
    result=nominator/denominator[0][0]
    return result

#main function
constants=np.random.rand(3)*1
lamda=calculate_lamda_0(constants[0],constants[1], constants[2])
param_change=3
iter_count=0
while(iter_count<500 and abs(param_change)>0.00001):
    delta_constant=calculate_delta_const(constants[0], constants[1], constants[2], lamda)
    constants_new=constants + delta_constant
    param_change=calculate_param_change(constants[0],constants[1],constants[2],constants_new[0],constants_new[1],constants_new[2], lamda, delta_constant)
    if(param_change>0.75):
        lamda=lamda/3
    elif(param_change<0.25):
        lamda=lamda*2
    else:
        lamda=lamda
    if(param_change>0):
        constants=constants_new
    iter_count+=1
print(iter_count)

#constants=[70500,80,850]

#OK:this part works fine. Problem lies in finding the constants.
xx=np.arange(0, 210, 0.05)
yy=breit_wigner(constants[0], constants[1], constants[2], xx)
plt.plot(xx,yy)
plt.scatter(df["x"], df["g"])
plt.suptitle("Breit-Wigner function for params a={:.2f}, b={:.2f}, c={:.2f}".format(constants[0], constants[1], constants[2]))
plt.savefig("Aufgabe6_2.png")