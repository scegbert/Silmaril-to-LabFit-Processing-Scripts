# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:03:37 2023

@author: scott
"""



import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')


import numpy as np
import pickle 
import pandas as pd

import matplotlib.pyplot as plt

import linelist_conversions as db

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()




#%% load in updated data

d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'


f = open(os.path.join(d_sceg,'df_sceg.pckl'), 'rb')
[df_sceg, _, _, _] = pickle.load(f)
f.close()

# df_sceg['quanta_index'] = df_sceg.vp + df_sceg.vpp


#%% load in HT 2020

d_HT = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2020\H2O.par'

df_HT2020 = db.par_to_df(d_HT)
df_HT2020.index = df_HT2020.index +1

#%% load in HT 2016

d_HT = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2016\H2O.par'

df_HT2016 = db.par_to_df(d_HT)
df_HT2016.index = df_HT2016.index +1

#%% load in HT 2010

d_HT = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITEMP 2010\H2O.par'

df_HT2010 = db.par_to_df(d_HT)
df_HT2010.index = df_HT2010.index +1

#%% load in Paul's data

d_paul = r'C:\Users\scott\Documents\1-WorkStuff\Labfit\working folder\paul nate og\PaulData_SD_Avgn_AKn2'

df_paul = db.labfit_to_df(d_paul, htp=False) # open paul database

df_paul['quanta_index'] = df_paul.quanta.str.replace('\s{2,}', ' ') # fix weird quanta formatting
df_paul.quanta_index = df_paul.quanta_index.str.slice_replace(0,2,'') # fix weird quanta formatting
df_paul.quanta_index = df_paul.quanta_index.str.slice_replace(23,24,'') # fix weird quanta formatting

#%% only the big ones (sw > 1e-21)


df_sw = df_HT2020[df_HT2020.sw > 1e-22].drop(columns = ['molec_id', 'local_iso_id' ,'gamma_air', 'gamma_self', 'n_air', 'delta_air', 'quanta'])


# merge in sceg
df_sw = pd.merge(df_sw, df_sceg[['sw','uc_sw']], how='inner', left_index=True, right_index=True, suffixes=('_2020', '_sceg'))
# df_sw = pd.merge(df_sw, df_sceg[['sw', 'quanta_index']], on='quanta_index', how='left')
df_sw = df_sw.rename(columns={'uc_sw':'uc_sw_sceg'})

# merge in paul
df_sw = pd.merge(df_sw, df_paul[['sw', 'uc_sw', 'quanta_index']], on='quanta_index', how='left')
df_sw = df_sw.rename(columns={'sw':'sw_paul', 'uc_sw':'uc_sw_paul'})

# merg in 2016
df_sw = pd.merge(df_sw, df_HT2016[['sw', 'quanta_index']], on='quanta_index', how='inner')
df_sw = df_sw.rename(columns={'sw':'sw_2016'})

# merge in HITRAN 2010
df_sw = pd.merge(df_sw, df_HT2010[df_HT2010.local_iso_id==1][['sw', 'quanta_index']], on='quanta_index', how='inner')
df_sw = df_sw.rename(columns={'sw':'sw_2010'})


df_sw['sw_ref'] = df_sw.iref.str[2:4]

#%% plots plots plots plots plots plots - everybody!


df_sw_plot = df_sw.copy()#[df_sw.sw_2020 > 1.2e-21]

ylim = [-0.12, 0.12]

# plt.figure()

# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_sceg-df_sw_plot.sw_2020)/df_sw_plot.sw_2020, 'x', markersize=10, label='Sceg')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2020-df_sw_plot.sw_2020)/df_sw_plot.sw_2020, '+', markersize=10, label='HITRAN 2020')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2016-df_sw_plot.sw_2020)/df_sw_plot.sw_2020, 'k1', markersize=10, label='HITRAN 2016')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_paul-df_sw_plot.sw_2020)/df_sw_plot.sw_2020, '2', markersize=10, label='Paul')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2010-df_sw_plot.sw_2020)/df_sw_plot.sw_2020, '3', markersize=10, label='HITEMP 2010')

# plt.xscale('log')

# plt.ylabel('SW (DATA - HITRAN 2020) / HITRAN 2020')
# plt.xlabel('SW HITRAN 2020')
# plt.ylim(ylim)
# plt.legend()


plt.figure()

plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_sceg-df_sw_plot.sw_2016)/df_sw_plot.sw_2016, 'x', markersize=10, label='Sceg')
plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2020-df_sw_plot.sw_2016)/df_sw_plot.sw_2016, '+', markersize=10, label='HITRAN 2020')
plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2016-df_sw_plot.sw_2016)/df_sw_plot.sw_2016, 'k1', markersize=10, label='HITRAN 2016')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_paul-df_sw_plot.sw_2016)/df_sw_plot.sw_2016, '2', markersize=10, label='Paul')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2010-df_sw_plot.sw_2016)/df_sw_plot.sw_2016, '3', markersize=10, label='HITEMP 2010')

plt.xscale('log')

plt.ylabel('SW (DATA - HITRAN 2016) / HITRAN 2016')
plt.xlabel('SW HITRAN 2020')
plt.ylim(ylim)
plt.legend()







# plt.figure()

# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_sceg-df_sw_plot.sw_2020), 'x', markersize=10, label='Sceg')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2020-df_sw_plot.sw_2020), '+', markersize=10, label='HITRAN 2020')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2016-df_sw_plot.sw_2020), 'k1', markersize=10, label='HITRAN 2016')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_paul-df_sw_plot.sw_2020), '2', markersize=10, label='Paul')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2010-df_sw_plot.sw_2020), '3', markersize=10, label='HITEMP 2010')

# plt.xscale('log')

# plt.ylabel('SW (DATA - HITRAN 2020)')
# plt.xlabel('SW HITRAN 2020')
# # plt.ylim(ylim)
# plt.legend()


# plt.figure()

# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_sceg-df_sw_plot.sw_2016), 'x', markersize=10, label='Sceg')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2020-df_sw_plot.sw_2016), '+', markersize=10, label='HITRAN 2020')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2016-df_sw_plot.sw_2016), 'k1', markersize=10, label='HITRAN 2016')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_paul-df_sw_plot.sw_2016), '2', markersize=10, label='Paul')
# plt.plot(df_sw_plot.sw_2020, (df_sw_plot.sw_2010-df_sw_plot.sw_2016), '3', markersize=10, label='HITEMP 2010')

# plt.xscale('log')

# plt.ylabel('SW (DATA - HITRAN 2016)')
# plt.xlabel('SW HITRAN 2020')
# # plt.ylim(ylim)
# plt.legend()









#%% plot wrt reference


plt.figure()

df_sw_plot1 = df_sw_plot[df_sw_plot.sw_ref == '63']

plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_sceg-df_sw_plot1.sw_2020)/df_sw_plot1.sw_2020, 'x', markersize=30, label='Sceg - Hodges')
plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_2016-df_sw_plot1.sw_2020)/df_sw_plot1.sw_2020, '1', markersize=30, label='HITRAN 2016 - Hodges')

df_sw_plot1 = df_sw_plot[df_sw_plot.sw_ref == '72']

plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_sceg-df_sw_plot1.sw_2020)/df_sw_plot1.sw_2020, 'x', markersize=10, label='Sceg - Conway')
plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_2016-df_sw_plot1.sw_2020)/df_sw_plot1.sw_2020, 'k1', markersize=10, label='HITRAN 2016 - Conway')

plt.hlines(0,1e-21, 2e-20)

plt.xscale('log')

plt.ylabel('SW (DATA - HITRAN 2020) / HITRAN 2020')
plt.xlabel('SW HITRAN 2020')
plt.ylim(ylim)
plt.legend()







plt.figure()

df_sw_plot1 = df_sw_plot[df_sw_plot.sw_ref == '63']

plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_sceg-df_sw_plot1.sw_2020), 'x', markersize=30, label='Sceg - Hodges')
plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_2016-df_sw_plot1.sw_2020), '1', markersize=30, label='HITRAN 2016 - Hodges')

df_sw_plot1 = df_sw_plot[df_sw_plot.sw_ref == '72']

plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_sceg-df_sw_plot1.sw_2020), 'x', markersize=10, label='Sceg - Conway')
plt.plot(df_sw_plot1.sw_2020, (df_sw_plot1.sw_2016-df_sw_plot1.sw_2020), 'k1', markersize=10, label='HITRAN 2016 - Conway')

plt.hlines(0,1e-21, 2e-20)

plt.xscale('log')

plt.ylabel('SW (DATA - HITRAN 2020)')
plt.xlabel('SW HITRAN 2020')
# plt.ylim(ylim)
plt.legend()




iso = df_sw.copy()
iso['2016_diff'] = (df_sw.sw_sceg-df_sw.sw_2016)/df_sw.sw_2016 * 100
iso['2020_diff'] = (df_sw.sw_sceg-df_sw.sw_2020)/df_sw.sw_2020 * 100















