#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:25:48 2022

@author: nilmasocastro
"""

import pandas as pd
import matplotlib.pyplot as plt
from sodapy import Socrata
import numpy as np
import geopandas as gpd
import seaborn as sns

#%%

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
client = Socrata("analisi.transparenciacatalunya.cat", None)

# Example authenticated client (needed for non-public datasets):
# client = Socrata(analisi.transparenciacatalunya.cat,
#                  MyAppToken,
#                  username="user@example.com",
#                  password="AFakePassword")

# First 8000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("qxev-y8x7", limit=7923)

#%%

# Convert to pandas DataFrame
data_df = pd.DataFrame.from_records(results)

#We remove the data of 2022 (since there is only 3 values of different RP and ABP (not correlated))
data_df = data_df.set_index('any')
data_df = data_df.drop('2022')
data_df = data_df.reset_index()


#Also erase the 'Divisió de Transport' RP since it has not analysis value
data_df = data_df.set_index('regi_policial_rp')
data_df = data_df.drop('Divisió de Transport')
data_df = data_df.reset_index()

#Since there is 2 RP Metropolitana Nord we need to change the name of these strings
data_df.loc['RP  Metropolitana Nord', 'regi_policial_rp'] = 'RP Metropolitana Nord'


#%%

print(data_df['regi_policial_rp'].value_counts())


#%%
'''
data_df['mitjana_temps_espera'] = pd.to_datetime(data_df['mitjana_temps_espera'])

data_df['mitjana_temps_espera'] = data_df['mitjana_temps_espera'].datatime.hour * 3600 + \
                                  data_df['mitjana_temps_espera'].datatime.minute * 60 + \
                                  data_df['mitjana_temps_espera'].datatime.second
'''

#%%
#Specific ABP
ABP = 'Alt Urgell'
single_ABP = data_df[data_df['rea_b_sica_policial_abp'] == 'ABP ' + ABP]

#ABP_Granollers_mes_visites = ABP_Granollers[['mes','nombre_de_visites']]

mes_visites = pd.DataFrame({'Mes':single_ABP.loc[:,'nom_mes'], 'Nombre visites':single_ABP.loc[:,'nombre_de_visites']})
mes_visites['Nombre visites'] = mes_visites['Nombre visites'].astype('int')
mes_visites = mes_visites.set_index('Mes')

#To plot the number of visits in one specific ABP
mes_visites.plot(kind='bar', title='ABP '+ABP, xlabel='Month', ylabel='Number of visits', 
                 rot=60, stacked=False, legend=False)







#%%
#Plot with circles of different sizes?
#Maybe the size of the circle is the mean time that every ABP takes (in a year).
#Flowing data (web or library)


#%%

#counts = data_df.value_counts()
##############################################

#Typo problem from 2017, where there is only 1 blankspace between RP and Metropolitana
#RP_Metropolitana_nord = data_df[data_df['regi_policial_rp'] == 'RP Metropolitana Nord']
#RP_Metropolitana_nord_2 = data_df[data_df['regi_policial_rp'] == 'RP  Metropolitana Nord']

#Also, from 'gener 2021' the month name is written all in lowercase







