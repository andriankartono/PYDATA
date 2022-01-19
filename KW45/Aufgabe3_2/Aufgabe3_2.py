'''
You'll have a bunch of datafiles including the yearly average
temperature for the country given in the file-name as an ISO
3-letter abbreviation from 1901 to 2012 (courtesy to the
worldbank)Now:
2.1 read in all the files in a dictionary structure, where the key is the
abbreviated country-code and the value is a list of all the data-lines
included in the file converted to float (you can assume that all the
lines in the files can be converted to float without error!, don't use a
numpy-array just yet!!!):
Output the dictionary to a json-file raw data.json with the
json.dump-function (help: import json; help(json.dump).
2.2 Now iterate over the datastructure from a and extract aggregated
data into a dictionary, where the keys are the 3-letter countrycodes
and the values are dictionaries of the following form:
1 >>> output_by_country [" DEU "]
2 { ' t_avg ': 8.475385244403567 , ' t_max_time ': 2000 ,
' t_max ': 10.041376113891602 , ' t_min_time ':
1940 , ' t_min ': 6.626019477844237}
calculate the aggregated data by converting the data to a numpy
array and calling the appropriate numpy functions on it. The
aggregated keys should be:
t avg the average temperature over the whole timeperiod
t max(min) time the year of the highest/lowest temperature
t max(min) the actual avg. temperature in the year of the
highest/lowest temperature
Output the dictionary to a json-file aggregated data.json with the
json.dump-function
'''
import pandas as pd
import json
import numpy as np
#should only be either glob or only os
import glob
import os

afp= open(r'D:\python\KW45\Aufgabe3_2\aggregated_data.json', 'w')
fp= open(r'D:\python\KW45\Aufgabe3_2\result3_2_1.json', 'w')
path=r'D:\python\KW45\Aufgabe3_2\tas_data'
filenames=glob.glob(path +'\*.csv')
data_by_country={}
output_by_country={}

for filename in filenames:
    df=pd.read_csv(filename, header=None)
    abbreviation=filename[-7:-4]
    data_by_country[abbreviation]=df.iloc[0:,0].values.tolist()
    numpy_array= df.iloc[0:,0].to_numpy()
    
    #calculate key values
    nested_dict={}
    nested_dict['t_avg']=int(np.average(numpy_array))
    nested_dict['t_max_time']=int(np.argmax(numpy_array)+1901)
    nested_dict['t_max']=int(np.amax(numpy_array))
    nested_dict['t_min_time']=int(np.argmin(numpy_array)+1901)
    nested_dict['t_min']=int(np.amin(numpy_array))
    
    #create a nested dictionary in output_by_country
    output_by_country[abbreviation]= nested_dict
    
#print(data_by_country['ABW'])
json.dump(data_by_country, fp)
json.dump(output_by_country, afp)


