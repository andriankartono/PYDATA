'''
Parallel to the strain gauge measurement, a pressure sensor was
continually recording the pressure (pressure sensor .csv). Write a
program which calculates the simple moving average
(windowsize=5) over this data and outputs to a file with an
accuracy of 2 decimals as as millivolts (ignore any empty lines from
the original file, and do an averaging of the first 5 datapoints as
well by continously increasing window-size with each datapoint)
'''

import pandas as pd
import matplotlib as mpl
import matplotlib . pyplot as plt

data=pd.read_csv(r'D:\python\KW43\aufgabe2_5\pressure_sensor.csv')
length=len(data)

data['WIKA_A10[mV]']= data['WIKA_A10-V'].rolling(5, min_periods=1).mean()
#data['WIKA_A10[mV]'].to_csv("check.csv")
#for row in range (0,5):
#    data.iat[row,1]=data.iloc[0:row+1,0].mean()

data['WIKA_A10[mV]']=data['WIKA_A10[mV]'].multiply(1000)
#data['WIKA_A10[mV]']=data['WIKA_A10[mV]'].multiply(1000).round(2)
#data['WIKA_A10[mV]'].to_csv(r'D:\python\KW43\aufgabe2_5\output2_5.csv', sep='\t', index=False)
data['WIKA_A10[mV]'].to_csv(r'D:\python\KW43\aufgabe2_5\output2_5.csv', sep='\t', index=False, float_format='%.2f')
data['WIKA_A10-V'].to_csv(r'D:\python\KW43\aufgabe2_5\test.csv', sep='\t', index=False)
print(data)

plt . plot ( range (len ( data['WIKA_A10-V'].multiply(1000) ) ) , data['WIKA_A10-V'].multiply(1000) )
plt . plot ( range (len ( data['WIKA_A10[mV]']) ) , data['WIKA_A10[mV]'])
plt.savefig(r'D:\python\KW43\aufgabe2_5\output2_5.png')
plt . show ()

