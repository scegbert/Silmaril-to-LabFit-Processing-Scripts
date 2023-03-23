r'''

HITRAN2 separate data by quantum assignment

creates different HITRAN files where each file is a different vibrational band

r'''



#%% -------------------------------------- load some libraries -------------------------------------- 

import numpy as np
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td # time doamain support
import linelist_conversions as db

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

from scipy.constants import speed_of_light

d_database = r"C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2020 by Quantum\\"

#%% -------------------------------------- setup wavenumber axis that spans large range ---------------- 

wvn = np.arange(6500,7800,200e6/speed_of_light/100)
wvl = 10000 / wvn

PL = 91.4 # pathlength in cm
P = 16 / 760 # pressure in atm
T = 1100 # T in K
y = 1

molecule = 'H2O'
molecule_id = 1


#%% -------------------------------------- separate the database by quantum vibrational assignments -------------------------------------- 

generate_q_model = False # do you want to save the databases? 
d_database_save = d_database + r"separated\\"

df_water = db.par_to_df(d_database + molecule + '.par')

df_water_v1v2v3 = df_water['quanta'].str.replace(' ', '').str[0:6]

h2o_quanta = []
h2o_quanta_count = []

for v, n in df_water_v1v2v3.value_counts().iteritems(): 
    
    if n>200 and v.find('-')==-1: # if they're positive (real) and there are at least 50 features
        print(v)
        h2o_quanta.append(v)
        h2o_quanta_count.append([v,n])
        
        if generate_q_model: 
            df_band = df_water.loc[df_water_v1v2v3 == v]
            db.df_to_par(df_band.reset_index(), par_name=molecule+'-'+v, save_dir=d_database_save)

h2o_quanta = ['002000', '101000', '200000', '021000', '120000', '111010', '012010', # all with over 200 transitions
             '210000', '031010', '210010', '040000', '050000', '130010', '111000',
             '130000', '121020', '300000', '041020', '220020', '201100', '050010',
             '031000', '102001', '300100', '060010', '121000', '300001', '102100']
    
h2o_quanta = ['101000', '200000', '021000', '002000','120000'] # keepers


#%% -------------------------------------- setup the combined model -------------------------------------- 

TD_water = np.zeros((len(h2o_quanta),(len(wvn)-1)*2))
trans_water = np.zeros((len(h2o_quanta),(len(wvn))))

# iterate through quantum assignment chunks for the first entry (hot water)
for j, v in enumerate(h2o_quanta): 
        
    if j == 0: pld.db_begin(d_database_save)

    print('generating model for ' + molecule + ' vibrational band ' + v[0:3] + '<-' + v[3:6])
    
    mod, pars = td.spectra_single_lmfit() 
    
    pars['mol_id'].value = molecule_id
    pars['shift'].vary = False
                        
    pars['pathlength'].set(value = PL, vary = False)
    pars['pressure'].set(value = P + P*np.random.rand()/1000, vary = False)
    pars['temperature'].set(value = T + T*np.random.rand()/1000, vary = True, min=250, max=3000)
               
    pars['molefraction'].set(value = y, vary = True)
    
    TD_water[j,:] = mod.eval(xx=wvn, params=pars, name=molecule+'-'+v)
    trans_water[j,:] = np.exp(-np.real(np.fft.rfft(TD_water[j,:])))
        
    
#%% -------------------------------------- convert to transmission and plot -------------------------------------- 

trans_water_all = np.exp(-np.sum(np.real(np.fft.rfft(TD_water,axis=1)) ,axis=0))

plt.figure(1)
# plt.plot(wvl, trans_water_all, label='all water')

plt.xlabel('Wavelength (um)')
plt.ylabel('Transmission')

for j, v in enumerate(h2o_quanta): 
    
    plt.figure(j+50)
    # plt.plot(wvl, trans_water_all)
    # plt.plot(wvl, trans_background_all)
    plt.plot(wvl,trans_water[j,:], label=v[0:3] + '<-' + v[3:6])
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Transmission')
    plt.legend()

    plt.figure(1)
    plt.plot(wvl,trans_water[j,:], label=v[0:3] + '<-' + v[3:6])

# plt.plot(wvl, trans_background_all, label='background')

plt.legend()

        


