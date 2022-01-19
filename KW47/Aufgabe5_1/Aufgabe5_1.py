'''
During the current Covid-epidemic, a lot of people have been
plotting a lot of (often meaningless) data. Nowadays everyone is
talking about the 7-day incidence of new cases and now you want
to do a plot of this. For that, get the tsv data for BY, NW, SN,
and TH from this github repo (tip, use the “raw-data”-button to
get a url which you can put into pd.read csv).
Then plot the “Cases Last Week”-column for each dataset vs. the
date (you can also shortcut setting the ticks-labels by specifying
the date-pandas-series for the x-data in the . plot-call). Set the
y-Axis to scale logarithmically.
Identify the global maximum and annotate it with an arrow
pointing a the maximum-point.
Additionally, in an inset, plot the total case number for the whole
of germany. You can create an inset plot with the following code
for main axes ax:
1 from mpl_toolkits . axes_grid1 . inset_locator import
inset_axes
2 inset_ax = inset_axes ( ax , " 100% ", " 100% ", # "100%"
means to fill the bbox fully
3 # location within the bounding box
4 loc 'lower left ',
5 # bounding box of the inset x0 , y0 ( lower
left corner ), width , heigth
6 bbox_to_anchor =( x0 , y0 , width , heigth ) ,
7 # coordinate system for the bounding box
( here probably : relative coordinates w.r.t. to
plot area )
8 bbox_transform = ax . transAxes )
'''
from os import sep
from matplotlib import legend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits . axes_grid1 . inset_locator import inset_axes

URL= "https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/master/data/de-states/de-state-{}.tsv"
bd_list=["BY", "NW", "SN", "TH"]
df_dict={}
maxim=0
maxim_in=None
axes=plt.subplot()
for bd in bd_list:
    complete_url= URL.format(bd)
    df_dict[bd]=pd.read_csv(complete_url, sep='\t', index_col="Date")
    df_dict[bd].index=pd.to_datetime(df_dict[bd].index)
    temp= df_dict[bd].plot(kind='line', y='Cases_Last_Week_Per_Million', logy=True, ax=axes, label='{}'.format(bd))
    temp_max=df_dict[bd]['Cases_Last_Week_Per_Million'].max()
    if int(temp_max)>int(maxim):
        maxim_in=bd
        maxim=temp_max

complete_df=pd.concat([df_dict["BY"]['Cases_Last_Week_Per_Million'], df_dict["NW"]['Cases_Last_Week_Per_Million'], df_dict["SN"]['Cases_Last_Week_Per_Million'], df_dict["TH"]['Cases_Last_Week_Per_Million']])
#complete_df.to_csv("test.csv", sep='\t')
x=complete_df.idxmax()
y=complete_df.max()

total_cases_df= pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/master/data/de-states/de-state-DE-total.tsv", sep='\t', index_col="Date")
total_cases_df.index=pd.to_datetime(total_cases_df.index)
inset_ax = inset_axes ( axes, width="100%", height="100%" ,bbox_to_anchor=(0.6, 0.1, 0.33, 0.15), bbox_transform=axes.transAxes)
inset_plot= total_cases_df.plot(kind='line', y='Cases_Last_Week_Per_Million', logy=True, ax=inset_ax, legend=None, xlabel=None)

axes.set_title('7-day incidence of Covid-cases')
axes.tick_params(axis='both', which='major', labelsize=6)
axes.annotate("Maximum n={} in {}\n @{}".format(y,maxim_in,x) ,xy=(x,y), xytext=(-100,-20) ,textcoords='offset points' ,arrowprops=dict(arrowstyle="->", connectionstyle='arc3'), fontsize=7)
axes.set_ylabel('n/week')

inset_ax.set_title('Incidence in Whole Germany', fontsize=10)
inset_ax.tick_params(axis='both', which='major', labelsize=4)
inset_ax.set_xlabel('')

plt.savefig("output5_1.png")


