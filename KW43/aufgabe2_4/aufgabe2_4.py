'''
Process the provided datafile (strain gauge rosette .csv,
<TAB>/\t-separated) of measurements with a
strain-gauge-rosette and calculate the two additionals columns for
the principal strains (for backgroundinfo and formulas look here
here; columns R2 X−m/m map to εX with X=(1,2,3) being
X=(a,b,c) respectively. Use e 1/2 instead of εmax/min for output
column names) Output should look like (8 decimal places only,
columns of the “csv”-style file separated by \t) (e.g. a
tab-character):
'''

import pandas as pd
import math

#file1 = open("D:/python/KW43/Output2_3.txt","w")
data= pd.read_csv(r"D:\python\KW43\aufgabe2_4\strain_gauge_rosette.csv", sep='\t')
data['e1']=''
data['e2']=''

for row in range(len(data)):
#for row in range(1):
    #define epsilon 1,2 and 3
    r1=data.iat[row,0]  
    r2=data.iat[row,1]
    r3=data.iat[row,2]
    #print(r3)
    
    #define the term under the squareroot to minimize mistake
    diff1=r1-r2
    diff2=r2-r3
    pow1=math.pow(diff1,2)
    pow2=math.pow(diff2,2)
    square=math.sqrt(2*(pow1+pow2))
    
    data.iat[row,3]=0.5*(r1+r3-square)
    data.iat[row,4]=0.5*(r1+r3+square)
    

data.astype(float).to_csv(r"D:\python\KW43\aufgabe2_4\output2_4.csv", sep='\t', index=False ,float_format='%.8f')
#print(data)
