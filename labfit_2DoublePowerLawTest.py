r'''

labfit2 double power law testing

process measurements one temperature at a time and then compile and compare


r'''



import subprocess

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as fit

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import linelist_conversions as db
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import labfithelp as lab

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

from copy import deepcopy

import pickle

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as colors


# %% define some dictionaries and parameters

d_type = 'air' # 'pure' or 'air'

if d_type == 'pure': 
    d_conditions = ['300 K _5 T', '300 K 1 T', '300 K 1_5 T', '300 K 2 T', '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
                    '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
                    '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
                    '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
                    '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T', '1300 K 16 T']
elif d_type == 'air': 
    d_conditions = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
                    '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
                    '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
                    '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
                    '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T', '1300 K 600 T']

# name in df, symbol for plotting, location of float (0 or 1) in INP file, number for constraint, acceptable uncertainty for fitting
props = {}
props['nu'] = ['nu', 'ν', 1, 23, 0.0015] 
props['sw'] = ['sw', '$S_{296}$', 2, 24, 0.09] # 9 % percent
props['gamma_air'] = ['gamma_air', 'γ air', 3, 25, 0.012] 
props['elower'] = ['elower', 'E\"', 4, 34, 200] # only floating this when things are weird
props['n_air'] = ['n_air', 'n air', 5, 26, 0.13]
props['delta_air'] = ['delta_air', 'δ air', 6, 27, 0.005]
props['n_delta_air'] = ['n_delta_air', 'n δ air', 7, 28, 0.13]
props['MW'] = ['MW', 'MW', 8, 29, 1e6]
props['gamma_self'] = ['gamma_self', 'γ self', 9, 30, 0.10]
props['n_self'] = ['n_self', 'n γ self', 10, 31, 0.13]
props['delta_self'] = ['delta_self', 'δ self', 11, 32, 0.005]
props['n_delta_self'] = ['n_delta_self', 'n δ self', 12, 33, 0.13]
props['beta_g_self'] = ['beta_g_self', 'βg self', 13, 35, 1e6] # dicke narrowing (don't worry about it for water, can't float with SD anyway)
props['y_self'] = ['y_self', 'y self', 14, 36, 1e6] # rosenkrantz line mixing (don't worry about this one either)
props['sd_self'] = ['sd_self', 'speed dependence', 15, 37, 0.10] # pure and air
props[False] = False # used with props_which2 option (when there isn't a second prop)

buffer = 2 # I added a cm-1 buffer to avoid weird chebyshev edge effects at bin edges
bin_breaks = [6500.2, 6562.8, 6579.7, 6599.5, 6620.6, 6639.4, 6660.2, 6680.1, 6699.6, 6717.9,
              6740.4, 6761.0, 6779.6, 6801.8, 6822.3, 6838.3 ,6861.4, 6883.2, 6900.1, 6920.2,
              6940.0, 6960.5, 6982.9, 7002.5, 7021.4, 7041.1, 7060.5, 7081.7, 7099.0, 7119.0, 
              7141.4, 7158.3, 7177.4, 7198.2, 7217.1, 7238.9, 7258.4, 7279.7, 7301.2, 7321.2, 
              7338.9, 7358.5, 7377.1, 7398.5, 7421.0, 7440.8, 7460.5, 7480.6, 7500.1, 7520.4,
              7540.6, 7560.5, 7580.5, 7600.0, 7620.0, 7640.0, 7660.0, 7720.0, 7799.8]
bin_names = ['B1',  'B2',  'B3',  'B4',  'B5',  'B6',  'B7',  'B8',  'B9',  'B10', 
             'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
             'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
             'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
             'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47', 'B48', 'B49', 'B50',
             'B51', 'B52', 'B53', 'B54', 'B55', 'B56', 'B57', 'B58']
if d_type == 'air': bin_names = [item + 'a' for item in bin_names] # append an a to the end of the bin name

bins = {} # dictionary (key is bin_names, entries are bin_breaks on either side)
for i in range(len(bin_names)):   
    bins[bin_names[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 
    if i == 0: bins[bin_names[i]][0] = 0
    elif i == len(bin_names)-1: bins[bin_names[i]][-1] = 0
bins['all'] = [-buffer, 6700, 7158.3, buffer] 

d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'
d_labfit_main = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit'

d_labfit_kp1 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP1'
d_labfit_kp2 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP2'
d_labfit_kp3 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP3'
d_labfit_kp4 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP4'
d_labfit_kp5 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP5'
d_labfit_kp6 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP6'
d_labfit_kp7 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP7'


d_sceg_save = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'



if d_type == 'pure': 
    base_name = 'p2020' + 'n_gam' # n update name
    d_cutoff_locations = d_labfit_main + '\\cutoff locations pure.pckl'

elif d_type == 'air': 
    base_name = 'p2020HTa' # 'p2020a_updated' 
    d_cutoff_locations = d_labfit_main + '\\cutoff locations air.pckl'

ratio_min_plot = -2 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
offset = 2 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

# %% run specific parameters and function executions


cutoff_s296 = 5E-24 

bin_name = 'B14a' # name of working bin (for these calculations)

d_labfit_kernal = d_labfit_kp4 # d_labfit_main # d_labfit_kp1




# d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files
d_old_holder = r'E:\water database\air water'
d_old = os.path.join(d_old_holder, bin_name, bin_name + '-000-og') # for comparing to original input files

# use_rei = True

# features_doublets_remove = []
# features_remove_manually = []

prop_which2 = False
prop_which3 = False

nudge_sd = True
features_reject_old = []


print('\n\n\n     ******************************************')
print('     *************** using bin {} ******************       '.format(bin_name))
if d_labfit_kernal == d_labfit_main: print('************** using MAIN Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp1: print('************** using KP #1 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp2: print('************** using KP #2 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp3: print('************** using KP #3 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp4: print('************** using KP #4 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp5: print('************** using KP #5 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp6: print('************** using KP #6 Labfit folder **************')
elif d_labfit_kernal == d_labfit_kp7: print('************** using KP #7 Labfit folder **************')
print('     ******************************************\n\n\n')




d_labfit_main = d_old_holder
features_new = None


# %% read in Labfit results

d_sceg = os.path.join(os.path.abspath(''),'data - sceg')

f = open(os.path.join(d_sceg,'df_sceg_pure.pckl'), 'rb')
[df_sceg_pure, _, df_HT2020_HT, _, _, _] = pickle.load(f)
f.close()

df_sceg_pure.loc[df_sceg_pure.uc_nu==0, 'uc_nu'] = 0.0015 
df_sceg_pure.loc[df_sceg_pure.uc_gamma_self==0, 'uc_gamma_self'] = 0.1
df_sceg_pure.loc[df_sceg_pure.uc_n_self==0, 'uc_n_self'] = 0.13
df_sceg_pure.loc[df_sceg_pure.uc_sd_self==0, 'uc_sd_self'] = 0.1 


df_sceg_pure['uc_nu_stat'] = df_sceg_pure.uc_nu.copy()

df_sceg_pure['uc_gamma_self_stat'] = df_sceg_pure.uc_gamma_self.copy()
df_sceg_pure['uc_n_self_stat'] = df_sceg_pure.uc_n_self.copy()
df_sceg_pure['uc_sd_self_stat'] = df_sceg_pure.uc_sd_self.copy()

df_sceg_pure['uc_delta_self_stat'] = df_sceg_pure.uc_delta_self.copy()
df_sceg_pure['uc_n_delta_self_stat'] = df_sceg_pure.uc_n_delta_self.copy()


which = (df_sceg_pure.uc_nu>-0.5)
df_sceg_pure.loc[which, 'uc_nu'] = np.sqrt(df_sceg_pure[which].uc_nu_stat**2 + 
                               (1.7E-4)**2)

    
# self parameters
which = (df_sceg_pure.uc_gamma_self>-0.5)
df_sceg_pure.loc[which, 'uc_gamma_self'] = np.sqrt((df_sceg_pure[which].uc_gamma_self_stat/df_sceg_pure[which].gamma_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2) * df_sceg_pure[which].gamma_self
which = (df_sceg_pure.uc_n_self>-0.5)
df_sceg_pure.loc[which, 'uc_n_self'] = np.sqrt((df_sceg_pure[which].uc_n_self_stat/df_sceg_pure[which].n_self)**2 + 
                                   (0.9645*df_sceg_pure[which].uc_gamma_self/df_sceg_pure[which].gamma_self)**2) * abs(df_sceg_pure[which].n_self)
which = (df_sceg_pure.uc_sd_self>-0.5)
df_sceg_pure.loc[which, 'uc_sd_self'] = np.sqrt((df_sceg_pure[which].uc_sd_self_stat/df_sceg_pure[which].sd_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2 + 0.039**2) * df_sceg_pure[which].sd_self

which = (df_sceg_pure.uc_delta_self>-0.5)
df_sceg_pure.loc[which, 'uc_delta_self'] = np.sqrt((df_sceg_pure[which].uc_delta_self_stat/df_sceg_pure[which].delta_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2 + #) * df_sceg_pure[which].delta_self
                                       (1.7E-4 / (0.021*df_sceg_pure[which].delta_self))**2) * abs(df_sceg_pure[which].delta_self)

which = (df_sceg_pure.uc_n_delta_self>-0.5)
df_sceg_pure.loc[which, 'uc_n_delta_self'] = np.sqrt((df_sceg_pure[which].uc_n_delta_self_stat/df_sceg_pure[which].n_delta_self)**2 + 
					   (0.9645*df_sceg_pure[which].uc_delta_self/df_sceg_pure[which].delta_self)**2) * abs(df_sceg_pure[which].n_delta_self)



# air water data
f = open(os.path.join(d_sceg,'df_sceg_air.pckl'), 'rb')
[df_sceg_air, _, _, _, _, _] = pickle.load(f)
f.close()

df_sceg_air = df_sceg_air.rename(columns={'sd_self':'sd_air', 'uc_sd_self':'uc_sd_air'})


df_sceg_air.loc[df_sceg_air.uc_gamma_self==0, 'uc_gamma_air'] = 0.012
df_sceg_air.loc[df_sceg_air.uc_n_self==0, 'uc_n_air'] = 0.13
df_sceg_air.loc[df_sceg_air.uc_sd_air==0, 'uc_sd_air'] = 0.1 


df_sceg_air['uc_gamma_air_stat'] = df_sceg_air.uc_gamma_air.copy()
df_sceg_air['uc_n_air_stat'] = df_sceg_air.uc_n_air.copy()
df_sceg_air['uc_sd_air_stat'] = df_sceg_air.uc_sd_air.copy()

df_sceg_air['uc_delta_air_stat'] = df_sceg_air.uc_delta_air.copy()
df_sceg_air['uc_n_delta_air_stat'] = df_sceg_air.uc_n_delta_air.copy()

   
# air parameters
which = (df_sceg_air.uc_gamma_air>-0.5)
df_sceg_air.loc[which, 'uc_gamma_air'] = np.sqrt((df_sceg_air[which].uc_gamma_air_stat/df_sceg_air[which].gamma_air)**2 + 
						   (0.0025)**2 + (0.029)**2 + (0.0082)**2) * df_sceg_air[which].gamma_air
which = (df_sceg_air.uc_n_air>-0.5)
df_sceg_air.loc[which, 'uc_n_air'] = np.sqrt((df_sceg_air[which].uc_n_air_stat/df_sceg_air[which].n_air)**2 + 
					   (0.9645*df_sceg_air[which].uc_gamma_air/df_sceg_air[which].gamma_air)**2) * abs(df_sceg_air[which].n_air)

which = (df_sceg_air.uc_sd_air>-0.5)
df_sceg_air.loc[which, 'uc_sd_air'] = np.sqrt((df_sceg_air[which].uc_sd_air_stat/df_sceg_air[which].sd_air)**2 + 
						   (0.0025)**2 + (0.029)**2 + (0.0082)**2 + 0.052**2) * df_sceg_air[which].sd_air

which = (df_sceg_air.uc_delta_air>-0.5)
df_sceg_air.loc[which, 'uc_delta_air'] = np.sqrt((df_sceg_air[which].uc_delta_air_stat/df_sceg_air[which].delta_air)**2 + 
						   (0.0025)**2 + (0.029)**2 + (0.0082)**2 + #) * df_sceg_air[which].delta_air
						   (1.7E-4 / (0.789*df_sceg_air[which].delta_air))**2) * abs(df_sceg_air[which].delta_air) # 0.789 atm = 600 Torr

which = (df_sceg_air.uc_n_delta_air>-0.5)
df_sceg_air.loc[which, 'uc_n_delta_air'] = np.sqrt((df_sceg_air[which].uc_n_delta_air_stat/df_sceg_air[which].n_delta_air)**2 + 
					   (0.9645*df_sceg_air[which].uc_delta_air/df_sceg_air[which].delta_air)**2) * abs(df_sceg_air[which].n_delta_air)


df_sceg_pure = df_sceg_pure[['nu','uc_nu','gamma_self','uc_gamma_self','n_self','uc_n_self','sd_self','uc_sd_self',
                             'delta_self','uc_delta_self','n_delta_self','uc_n_delta_self']]

df_sceg_air = df_sceg_air[['gamma_air','uc_gamma_air','n_air','uc_n_air','sd_air','uc_sd_air',
                           'delta_air','uc_delta_air','n_delta_air','uc_n_delta_air',
                           'quanta','local_iso_id','vp','vpp','Jp','Kap','Kcp','Jpp','Kapp','Kcpp','m','doublets', 
                           'ratio_300','ratio_500','ratio_700','ratio_900','ratio_1100','ratio_1300']]


df_sceg = pd.merge(df_sceg_pure, df_sceg_air, on='index')

df_HT2020_HT = df_HT2020_HT[['gamma_self','gamma_air','n_air','delta_air','ierr','iref']]
df_sceg_align, df_HT2020_HT_align = df_sceg.align(df_HT2020_HT, join='inner', axis=0)

df_sceg = pd.merge(df_sceg_align, df_HT2020_HT_align, left_index=True, right_index=True, suffixes=('', 'HT'))


#%% prepare transitions we will use to explore DPL


bin_names_test = ['B10', 
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
                  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
                  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47']

d_types = ['pure', 'air']

# put both air and pure items in the bins dict
if bin_names[0][-1] == 'a' and 'B1' not in bins.keys(): 
    bin_names2 = [item[:-1] for item in bin_names]
    
    for i in range(len(bin_names2)):   
        bins[bin_names2[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 
        if i == 0: bins[bin_names2[i]][0] = 0
        elif i == len(bin_names2)-1: bins[bin_names2[i]][-1] = 0

elif bin_names[0][-1] != 'a' and 'B1a' not in bins.keys(): 
    bin_names2 = [item+'a' for item in bin_names]
    
    for i in range(len(bin_names2)):   
        bins[bin_names2[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 
        if i == 0: bins[bin_names2[i]][0] = 0
        elif i == len(bin_names2)-1: bins[bin_names2[i]][-1] = 0


features_strong =  [[7094], # features with a clean absorption peak at all conditions, see excel file for more details
                    [8051, 8085, 8220],
                    [8450, 8793],
                    [9298, 9566],
                    [10165],
                    [10827],
                    [11063, 11124],
                    [12001, 12136],
                    [12647, 12837, 12952, 13048],
                    [13848, 13867, 13873, 13886, 13950],
                    [14045, 14060, 14459, 14509, 14740, 14742, 14817],
                    [15077, 15097, 15194, 15487, 15535, 15596],
                    [15812, 16182, 16259, 16295, 16467, 16551, 16558, 16572],
                    [16735, 17112, 17170, 17290, 17295, 17300, 17339, 17383, 17423, 17428],
                    [17677, 17856, 17955, 18011],
                    [18339, 18398, 18406, 18478, 18555, 18611, 18974, 19055],
                    [19207, 19281, 19333, 19339, 19346, 19398, 19406, 19463, 19707],
                    [20283, 20315, 20320, 20349, 20429, 20550, 20835],
                    [21075, 21189, 21361, 21430, 21484],
                    [21873, 21929, 21991, 22025, 22060, 22337, 22422, 22431, 22455],
                    [22562, 22611, 22707, 22855, 22893, 23161, 23200, 23250, 23360, 23378],
                    [23499, 23615, 24128],
                    [24324, 24484, 24506, 24605, 24802],
                    [25176, 25210, 25233, 25495, 25502, 25534, 25695, 25732],
                    [26046, 26067, 26134, 26190, 26365, 26463, 26578, 26611],
                    [26691, 27034, 27107, 27181, 27194, 27227, 27244, 27268, 27282, 27307, 27334, 27348, 27442, 27475],
                    [27622, 27698, 27730, 27839, 28117, 28152, 28187],
                    [28492, 28543, 28805, 28824, 29054, 29069],
                    [29433, 29524, 29550, 29743, 29840, 29937, 30022, 30272],
                    [30697, 30781, 30851, 30999, 31006],
                    [31330, 31379, 31504, 31555, 31565, 31680, 31761, 32034],
                    [32116, 32145, 32215, 32308, 32453, 32506, 32634, 32767, 32805, 32870],
                    [32958, 33197, 33209, 33330, 33347, 33596, 33603, 33612],
                    [33706, 33757, 33786, 33801, 33811, 33956, 34111, 34228, 34360],
                    [34403, 34413, 34508, 34537, 34617, 34662, 34834, 34892, 34917, 34962, 35005, 35036],
                    [35195, 35251, 35348, 35597],
                    [35745, 35926, 36029],
                    [36383]]




features_strong_flat_all = [item for sublist in features_strong for item in sublist]
features_strong_flat = features_strong_flat_all.copy()


T_conditions = list(dict.fromkeys([i.split()[0] for i in d_conditions]))

df_sceg = df_sceg[df_sceg.index.isin(features_strong_flat_all)]



d_whiches = ['self', 'air']
extra_params_base = ['gamma','sd','delta', 'iter'] # self and air width, SD, and shift
extra_params = ['nu_300', 'uc_nu_300']

for d_which in d_whiches: 
    for param in extra_params_base: 
        for T_iter in T_conditions:
            
            name = param + '_' + d_which + '_' + T_iter
            
            extra_params.append(name)
            
            if param != 'iter': 
                extra_params.append('uc_' + name)
    
    

df_sceg = pd.merge(df_sceg, df_sceg.reindex(columns=extra_params), left_index=True, right_index=True)


bin_names_test2 = bin_names_test.copy()
features_strong2 = features_strong.copy()
d_types2 = d_types.copy()

T_conditions2 = T_conditions.copy()


#%%  update parameters function 


def update_df(df_output, df_input, features, d_which, T_iter, i_labfit):
    # extract updated values and compile 
    for feat in features: 
        
        for i_p, param in enumerate(extra_params_base): 
    
            name_df = param + '_' + d_which + '_' + T_iter
            name_labfit = param + '_' + d_which
            
            if param == 'iter': 
                
                df_output.loc[feat, name_df] = i_labfit
                
            else: 
                
                if name_labfit == 'sd_air':
                    name_labfit = 'sd_self' # labfit calls everything SD self
                    
                df_output.loc[feat, name_df] = df_input.loc[feat][name_labfit]
                df_output.loc[feat, 'uc_'+name_df] = df_input.loc[feat]['uc_'+name_labfit]
                                 
                
                if name_df == 'gamma_self_300': 
        
                    df_output.loc[feat, 'nu_300'] = df_input.loc[feat]['nu']
                    df_output.loc[feat, 'uc_nu_300'] = df_input.loc[feat]['uc_nu'] 
                    
    return df_output


stop = DPL_processing_next


#%% perform DPL calculations


testing_one = False

# skip to a specific transition and condition
if testing_one: 
    
    feat = 12136
    T_target = '900'
    d_types = d_types2 # ['pure']  # not to run pure first to get to air
    
    bin_names_test = [bin_names_test2[[i for i, sublist in enumerate(features_strong2) if feat in sublist][0]]]
    features_strong = [[feat]]

    if T_target == '300': T_conditions = ['300']
    else: T_conditions = ['300',T_target]
    
    df_feat1 = None
    df_feat2 = None
    df_feat3 = None
    df_feat4 = None

    print('\n\n {}    {}'.format(feat, T_target))



iter_limit = 10 # iterate labfit up to X times



d_labfit_kernal = d_labfit_kp1
d_old_all = r'E:\water database' # for comparing to original input files

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)



for i_bin, bin_name in enumerate(bin_names_test):

    features = features_strong[i_bin]
            
    for i_type, d_type in enumerate(d_types): 
        
        if d_type == 'pure': 
            d_which = 'self'
            
        elif d_type == 'air': 
            d_which = 'air'    
            bin_name+='a'
        
        # get most up-to-date REI file into Labfit
        d_load_folder = os.path.join(d_old_all, d_type+' water')
        [_, use_which] = lab.newest_rei(os.path.join(d_load_folder, bin_name), bin_name)
        d_load_file = os.path.join(d_load_folder, bin_name, use_which)
        
        inp_saved = open(d_load_file, "r").readlines()
        
        num_asc = int(inp_saved[0].split()[2])
        num_features = int(inp_saved[0].split()[3])
         
        line_first_feature = lines_main_header + num_asc * lines_per_asc # all of the header down to the spectra
        line_first_constraint = line_first_feature + num_features * lines_per_feature
        
        # remove all constraints and unfloat all features
        inp_saved[1] = inp_saved[1][:26] + '      0      0' + inp_saved[1][40:] # remove all constraints from file header
        inp_saved = inp_saved[:line_first_constraint] # remove all constraints from bottom of file
        
        for i in range(num_features): 
            
            inp_saved[line_first_feature+i*lines_per_feature+2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n' # unfloat everything
        
        inp_header = inp_saved[:lines_main_header]
        inp_features = inp_saved[line_first_feature:]
        
        # extract updated values and compile into dict/list
        [T, P, wvn, trans_og, res_og, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_load_folder, bins, bin_name, og=True) # <-------------------

        d_og = os.path.join(d_old_all, d_type+' water', bin_name, bin_name + '-000-og') # for comparing to original input files
        df_calcs = lab.information_df(False, bin_name, bins, cutoff_s296, T, d_old=d_og, d_load=d_load_file[:-4]) # <-------------------

        for feat in features: 
            
            # set all temperature dependences to 0 
            i_guess = feat*lines_per_feature
            
            if int(inp_features[i_guess].split()[0]) != feat: 
                i_guess = lab.floated_line_moved(i_guess+2, feat, inp_features, lines_per_feature)
                i_guess -=2
                
            inp_features[i_guess] = inp_features[i_guess][:65] + ' 0.0000 ' + inp_features[i_guess][73:84] + ' 0.00000000 ' + inp_features[i_guess][96:]
            inp_features[i_guess+1] = inp_features[i_guess+1][:11] + ' 0.0000 ' + inp_features[i_guess+1][19:30] + ' 0.00000000 ' + inp_features[i_guess+1][42:]
    
        
        for i_T, T_iter in enumerate(T_conditions): 
            
            for i_meas, meas_condition in enumerate(d_conditions): 
                            
                # remove all measurement files except the ones we're investigating (T_iter)
                iter_asc = 0
                
                for i_asc in range(num_asc): 
                    
                    line_asc = lines_main_header+i_asc*lines_per_asc
                    
                    T_asc = inp_saved[line_asc].split()[0].replace('_', ' ').split('.')[0].split()[1]
                                    
                    if T_asc == T_iter: 
                        
                        if iter_asc == 0: 
                            iter_asc+=1
                            inp_asc = inp_saved[line_asc:line_asc+lines_per_asc] # for first (and/or only) instance
                            
                        else: 
                            iter_asc+=1
                            inp_asc.extend(inp_saved[line_asc:line_asc+lines_per_asc]) # if multiple instances of this ASC file
                        
            # update nu for all conditions that aren't 300 K, pure H2O
            if (T_iter == '300' and d_type == 'pure') is False:
                
                for feat in features: 
                                                
                    # wavenumber as string
                    nu_str = '{:.7f}'.format(df_sceg.nu_300[feat]) 
                    
                    # find the feature
                    i_guess = feat*lines_per_feature
                    
                    if int(inp_features[i_guess].split()[0]) != feat: 
                        i_guess = lab.floated_line_moved(i_guess+2, feat, inp_features, lines_per_feature)
                        i_guess -=2
                        
                    inp_features[i_guess] = inp_features[i_guess][:15] + nu_str + inp_features[i_guess][27:]
                    
                    # update width and shift to match values obtained for that conditions
                    if d_type == 'air': 
 
                        g_self_T = '{:.5f}'.format(df_sceg.loc[feat, 'gamma_self_'+T_iter])
                        d_self_T = '{:.7f}'.format(df_sceg.loc[feat, 'delta_self_'+T_iter])
                        if d_self_T[0] != '-': d_self_T = ' '+d_self_T
                        
                        inp_features[i_guess+1] = '   ' + g_self_T + inp_features[i_guess+1][10:19] + d_self_T + inp_features[i_guess+1][29:]
            
            # document number of files included and save INP            
            inp_header[0] = inp_header[0][:25] + '    {}  '.format(iter_asc) + inp_header[0][32:]
            
            inp_updated = inp_header.copy()
            inp_updated.extend(inp_asc)
            inp_updated.extend(inp_features)
            
            open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
            
            print('\n************************            {}          {}        {}\n'.format(bin_name, d_type, T_iter))
            
            # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
            if T_iter == '300' and d_type == 'pure': 
                lab.float_lines(d_labfit_kernal, bin_name, features, props['nu'], 'inp_new', []) 
            
            lab.float_lines(d_labfit_kernal, bin_name, features, props['gamma_'+d_which], 'inp_new', []) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['sd_self'], 'inp_new', []) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['delta_'+d_which], 'inp_new', []) 
            
            
            
            # run labfit for all transitions in this bin
            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
            
            
            # if that worked
            if feature_error is None: 
                
                # push updated values to df_sceg
                i_labfit = 0
                df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                df_sceg = update_df(df_sceg, df_calcs, features, d_which, T_iter, i_labfit)
                
                while feature_error is None and i_labfit < iter_limit: 
                    i_labfit+=1 
                    print(i_labfit)
                    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True, time_limit=30) 
                    
                    # keep iterating and pushing updated values to df_sceg
                    if feature_error is None: 
                        df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                        df_sceg = update_df(df_sceg, df_calcs, features, d_which, T_iter, i_labfit)
            
            
            # if that didn't work
            if feature_error is not None:  
                
                # go feature by feature
                for feat in features: 
                    
                    open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
                    
                    print('\n            trying one at a time, currently on {}'.format(feat))                                    
                    
                    # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
                    if T_iter == '300' and d_type == 'pure': 
                        lab.float_lines(d_labfit_kernal, bin_name, feat, props['nu'], 'inp_new', []) 
                    
                    lab.float_lines(d_labfit_kernal, bin_name, feat, props['gamma_'+d_which], 'inp_new', []) 
                    lab.float_lines(d_labfit_kernal, bin_name, feat, props['sd_self'], 'inp_new', []) 
                    lab.float_lines(d_labfit_kernal, bin_name, feat, props['delta_'+d_which], 'inp_new', []) 
                    
                    # run labfit
                    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) 

                    # if that worked
                    if feature_error is None: 
                        
                        # push updated values to df_sceg
                        i_labfit = 0

                        df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                        df_sceg = update_df(df_sceg, df_calcs, features, d_which, T_iter, i_labfit)
                        
                        while feature_error is None and i_labfit < iter_limit: 
                            i_labfit+=1 
                            print(i_labfit)
                            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True, time_limit=30) 
                            
                            # keep iterating and pushing updated values to df_sceg
                            if feature_error is None: 
                                df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                                df_sceg = update_df(df_sceg, df_calcs, features, d_which, T_iter, i_labfit)
                    
    
    
    
    f = open(os.path.join(d_sceg_save,'DPL results.pckl'), 'wb')
    pickle.dump([df_sceg, T_conditions, features_strong], f)
    f.close()               
    
    df_sceg.to_csv('DPL results.csv', float_format='%g')


#%% load in the DPL data


# for param in extra_params: 
#     df_sceg[param] = df_sceg2[param]


f = open(os.path.join(d_sceg_save,'DPL results.pckl'), 'rb')
[df_sceg, T_conditions, features_strong, features_doublets] = pickle.load(f)
f.close()     

types_RMS = ['labfit', 'HT', 'SPL', 'DPL']  
d_whiches = ['self', 'air']
extra_params_base = ['gamma','sd','delta'] # self and air width, SD, and shift

# RMS for all values
columns_RMS = []
for d_which in d_whiches: 
    for param in extra_params_base: 
        for type_RMS in types_RMS:
            
            name = 'RMS_' + type_RMS + '_' + param + '_' + d_which
            columns_RMS.append(name)

# check for over conditioning for only scipy calculated values
for d_which in d_whiches: 
    for param in extra_params_base: 
        for type_RMS in types_RMS[2:]: 
            
            name = 'condition_' + type_RMS + '_' + param + '_' + d_which
            columns_RMS.append(name)

df_sceg = pd.merge(df_sceg, df_sceg.reindex(columns=columns_RMS), left_index=True, right_index=True)

T_conditions = [float(T) for T in T_conditions]
T_conditions = np.asarray(T_conditions)


type_name = 'air' 

type_names = {'pure':0,
              'air':1}
i_type = type_names[type_name]


prop_name = 'sd'

prop_names = {'gamma':0,
              'sd':2, 
              'delta':4, 
              'nu':6}

i_prop = prop_names[prop_name]


#%% fit the data for n values

def SPL(T, c, n): 
    
    return c*(296/T)**n

def SPLoff(T, c1, c2, n): 

    return c1*(296/T)**n + c2
    
def DPL(T, c1, n1, c2, n2): 
    
    return c1*(296/T)**n1 + c2*(296/T)**n2


please = stophere_fullyloaded

#%% plots for Bob

plot_aw = False

T_smooth = np.linspace(T_conditions[0]-110, T_conditions[-1]+110, num=500)
df_quanta = db.labfit_to_df('E:\water database\pure water\B1\B1-000-HITRAN')

T_unc_pure = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 19/1288])
T_unc_air = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 20/1288])

P_unc_pure = 0.0027
P_unc_air = 0.0025
y_unc_air = 0.029

colors_fits = ['lime','darkorange','blue', 'red']



data_labfit = ['n_self', 
               'n_delta_self', 
               'sd_self', 
               'n_air', 
               'n_delta_air', 
               'sd_air']

data_HT = [['gamma_self','n_air'], 
           False, 
           False, 
           ['gamma_air', 'n_air'], 
           ['delta_air', False], 
           False]
unc_HITRAN = {'gamma_self':3,
              'gamma_air':2,
              'n_air':4,
               'delta_air':5}

HT_errors = {'0': 0, # '0 (unreported)', 
             '1': 0, # '1 (default)', 
             '2': 0, # '2 (average)', 
             '3': 0, # '3 (over 20%)', 
             '4': 0.2, # '4 (10-20%)', 
             '5': 0.1, # '5 (5-10%)', 
             '6': 0.05, # '6 (2-5%)', 
             '7': 0.02, # '7 (1-2%)', 
             '8': 0.01} # '8 (under 1%)'}

name_plot = ['Self-Width\nγ$_{self}$ [cm$^{-1}$/atm]', 
             'Self-Shift\nδ$_{self}$ [cm$^{-1}$/atm]', 
             'SD Self-Width\nγ$_{SD,self}$  [cm$^{-1}$/atm]',
             'Air-Width\nγ$_{air}$ [cm$^{-1}$/atm]', 
             'Air-Shift\nδ$_{air}$ [cm$^{-1}$/atm]', 
             'SD Air-Width\nγ$_{SD,air}$  [cm$^{-1}$/atm]',
             'a$_{w,air}$\n(γ$_{SD,air}$ / γ$_{air}$)', 
             'a$_{w,self}$\n(γ$_{SD,self}$ / γ$_{self}$)']


for i_feat, feat in enumerate(df_sceg.index): 
    
    feat = int(feat)
    
    df_feat = df_sceg.loc[feat]   
    
    
    pure_T_index = [index for index in df_feat.index if '_self_' in index]
    air_T_index = [index for index in df_feat.index if '_air_' in index]
    
    # confirm all values are floated and not doublets (overly restrictive for now, will ease up later)
    if ~np.any(df_feat[pure_T_index]==-1) or ~np.any(df_feat[air_T_index]==-1): 

        df_gamma_self = df_feat[df_feat.index.str.startswith('gamma_self_')]
        df_uc_gamma_self = df_feat[df_feat.index.str.startswith('uc_gamma_self_')]
        
        
        df_uc_gamma_self[:] = np.sqrt((df_uc_gamma_self.to_numpy(float)/df_gamma_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_gamma_self
        
        df_sd_self = df_feat[df_feat.index.str.startswith('sd_self_')]
        df_uc_sd_self = df_feat[df_feat.index.str.startswith('uc_sd_self_')]
        df_uc_sd_self[:] = np.sqrt((df_uc_sd_self.to_numpy(float)/df_sd_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_sd_self # droped 3.9% due to TD of SD
        
        df_delta_self = df_feat[df_feat.index.str.startswith('delta_self_')]
        df_uc_delta_self = df_feat[df_feat.index.str.startswith('uc_delta_self_')]
        df_uc_delta_self[:] = np.sqrt((df_uc_delta_self.to_numpy(float)/df_delta_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2 + 
                                      (1.7E-4 / (0.021*df_delta_self.to_numpy(float)))**2) * abs(df_delta_self)
                                      
        
        df_gamma_air = df_feat[df_feat.index.str.startswith('gamma_air_')]
        df_uc_gamma_air = df_feat[df_feat.index.str.startswith('uc_gamma_air_')]
        df_uc_gamma_air[:] = np.sqrt((df_uc_gamma_air.to_numpy(float)/df_gamma_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_gamma_air
                
        df_sd_air = df_feat[df_feat.index.str.startswith('sd_air_')]
        df_uc_sd_air = df_feat[df_feat.index.str.startswith('uc_sd_air_')]
        df_uc_sd_air[:] = np.sqrt((df_uc_sd_air.to_numpy(float)/df_sd_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_sd_air # droped 5.2
        
        df_delta_air = df_feat[df_feat.index.str.startswith('delta_air_')]
        df_uc_delta_air = df_feat[df_feat.index.str.startswith('uc_delta_air_')]
        df_uc_delta_air[:] = np.sqrt((df_uc_delta_air.to_numpy(float)/df_delta_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 +  (y_unc_air)**2 +
                                      (1.7E-4 / (0.789*df_delta_air.to_numpy(float)))**2) * abs(df_delta_air)
        
        
        
        if plot_aw: fig, axs = plt.subplots(4,2, figsize=(15, 10), sharex = 'col', gridspec_kw={'hspace':0.01, 'wspace':0.25}) 
        else: fig, axs = plt.subplots(3,2, figsize=(15, 10), sharex = 'col', gridspec_kw={'hspace':0.01, 'wspace':0.25}) 
        
        fig.subplots_adjust(top=0.95)
       
        quanta = df_feat.quanta.split()
        try: kcJ = float(quanta[9])/float(quanta[11])
        except: kcJ = 0.
               
        title = '{}{}{} ← {}{}{}      {}$_{}$$_{}$ ← {}$_{}$$_{}$      Kc" / J" = {:.3f}'.format(quanta[0],quanta[1],quanta[2],quanta[3],quanta[4],quanta[5],
                                                                        quanta[6],quanta[7],quanta[8],quanta[9],quanta[10],quanta[11], kcJ)
        
        if df_feat.doublets != []: title = '** Overlapping Doublet Transition ** ' + title 
        
        plt.suptitle(title)
        
        RMS_params = 9
        RMS_extra = 3

        RMS_iter = np.empty((1,RMS_extra + RMS_params*2))
        RMS_iter[:] = np.nan
        
        RMS_iter[0,0] = feat
        RMS_iter[0,1] = float(quanta[6]) # J"
        RMS_iter[0,2] = kcJ # Kc / J"
        
        
        data_plots = [[df_gamma_self, df_uc_gamma_self], 
                      [df_delta_self, df_uc_delta_self], 
                      [df_sd_self, df_uc_sd_self], 
                      [df_gamma_air, df_uc_gamma_air], 
                      [df_delta_air, df_uc_delta_air], 
                      [df_sd_air, df_uc_sd_air]]
                                
        for i_plot, [data, uc_data] in enumerate(data_plots): 
            
            i_1 = i_plot - 3*(i_plot//3)
            i_2 = i_plot//3            
            
            if (((i_2 == 0)&(~np.any(df_feat[pure_T_index]==-1))&(max(df_uc_sd_self)<0.1)) or
                ((i_2 == 1)&(~np.any(df_feat[air_T_index]==-1))&(max(df_uc_sd_air)<0.1))): 
                
                if data_labfit[i_plot][:2] == 'sd': 
                    
                    if plot_aw: 
    
                        # plot SD aw data
                        axs[i_1,i_2].plot(T_conditions, data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                        axs[i_1,i_2].errorbar(T_conditions, data, uc_data, color='k', fmt='none', capsize=5, zorder=10)
                        
                        if df_feat['uc_' + data_labfit[i_plot]] != -1: 
                            
                            y_center = np.ones_like(T_smooth) * df_feat[data.index[0][:-4]]
                            axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[0], label='Labfit (constant)')
                            
                            axs[i_1,i_2].set_ylabel(name_plot[-1*(1+i_2)])
                            axs[i_1,i_2].legend()
                            
                        i_1+=1
    
                    # plot SD gamma data
                    axs[i_1,i_2].plot(T_conditions, data*df_feat[df_feat.index.str.startswith('gamma'+data_labfit[i_plot][2:]+'_')].to_numpy(float),
                                      color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                    axs[i_1,i_2].errorbar(T_conditions, data*df_feat[df_feat.index.str.startswith('gamma'+data_labfit[i_plot][2:]+'_')].to_numpy(float),
                                          uc_data, color='k', fmt='none', capsize=5, zorder=10)
    
                
                else:     
                    
                    # plot the data (non-SD)
                    axs[i_1,i_2].plot(T_conditions, data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                    axs[i_1,i_2].errorbar(T_conditions, data, uc_data, color='k', fmt='none', capsize=5, zorder=10)
    
                    
                # overlay HITRAN prediction
                if data_HT[i_plot]: 
                    
                    base = df_feat[data_HT[i_plot][0] + 'HT']
                    uc_base_val = df_feat.ierr[unc_HITRAN[data_HT[i_plot][0]]]
                    uc_base = base * HT_errors[uc_base_val]
                    
                    if data_HT[i_plot][1]:
                        n = df_feat[data_HT[i_plot][1] + 'HT']
                        uc_n_val = df_feat.ierr[unc_HITRAN[data_HT[i_plot][1]]]
                        uc_n = n * HT_errors[uc_n_val]
                        
                    else: 
                        n=0
                        uc_n=0
                    
                    y_center = SPL(T_smooth, base, n)
                    y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                      SPL(T_smooth, base+uc_base, n-uc_n), 
                                      SPL(T_smooth, base-uc_base, n+uc_n),
                                      SPL(T_smooth, base-uc_base, n-uc_n)])
                    y_max = np.max(y_unc,axis=0)
                    y_min = np.min(y_unc,axis=0)
                    
                    
                    if data_HT[i_plot][0] == 'gamma_self': 
                        axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$ (using n$_{{{}}}$)'.format(base,n,'air'))
                    elif data_HT[i_plot][0] == 'gamma_air': 
                        axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                    elif data_HT[i_plot][0] == 'delta_air': 
                        axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f} (no TD)'.format(base,n))
                    
                    axs[i_1,i_2].fill_between(T_smooth, y_min, y_max, color=colors_fits[3], alpha=.2)
                    df_sceg.loc[feat, 'RMS_HT_'+data.index[0][:-4]] = np.sqrt(np.sum((SPL(T_conditions, base, n) - data)**2))
                        
                
                # overlay Labfit prediction (if there is one)
                if df_feat['uc_' + data_labfit[i_plot]] != -1: 
                    
                    if data_labfit[i_plot][:2] == 'sd':  
                       
                        base = df_feat[data.index[0][:-4]] * df_feat['gamma'+data_labfit[i_plot][2:]]
                        uc_base = df_feat['uc_'+data.index[0][:-4]]
                        
                        n = df_feat['n'+data_labfit[i_plot][2:]]
                        uc_n = df_feat['uc_n'+data_labfit[i_plot][2:]]
                        
                        data *= data_plots[3*i_2][0].to_numpy(float)
                        
                    else:                       
                        
                        base = df_feat[data.index[0][:-4]]
                        uc_base = df_feat['uc_'+data.index[0][:-4]]
                        
                        n = df_feat[data_labfit[i_plot]]
                        uc_n = df_feat['uc_'+data_labfit[i_plot]]
                    
                    y_center = SPL(T_smooth, base, n)
                    y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                    SPL(T_smooth, base+uc_base, n-uc_n), 
                                    SPL(T_smooth, base-uc_base, n+uc_n),
                                    SPL(T_smooth, base-uc_base, n-uc_n)])
                    y_max = np.max(y_unc,axis=0)
                    y_min = np.min(y_unc,axis=0)
                    
                    if data_labfit[i_plot][:2] == 'sd': axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[0], label='Labfit (from a$_w$), {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                    else: axs[i_1,i_2].plot(T_smooth, y_center, color=colors_fits[0], label='Labfit {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                    axs[i_1,i_2].fill_between(T_smooth, y_min, y_max, color=colors_fits[0], alpha=.2)
                
                    df_sceg.loc[feat, 'RMS_labfit_'+data.index[0][:-4]] = np.sqrt(np.sum((SPL(T_conditions, base, n) - data)**2))
                    
                    
                
                # fit the data using DPL (weighted by Labfit uncertainties)
                solving=True
                i_solve=-1
                p0s = [[2*base, n, -2*base, -1*n], [base/2, n, base/2, n]]
                while solving:
                    i_solve+=1
                    try: 
                        p_DPL, cov = fit(DPL, T_conditions, data, p0=p0s[i_solve], maxfev=5000, sigma=uc_data.to_numpy(float))
                        p_err = np.sqrt(np.diag(cov))
    
                        solving=False # you solved it!
                        axs[i_1,i_2].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], label='DPL {:.3f}(T/T)$^{{{:.2f}}}$+{:.3f}(T/T)$^{{{:.2f}}}$'.format(p_DPL[0],p_DPL[1],p_DPL[2],p_DPL[3]))
                        
                        # calculate uncertainties (too many Infs and NANS - haven't been using much)
                        y_unc = np.array([DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                          DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                          DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3])])
                        y_max = np.max(y_unc,axis=0)
                        y_min = np.min(y_unc,axis=0)
                        # if np.all(np.isfinite(y_max)) & np.all(np.isfinite(y_min)): 
                            # axs[i_1,i_2].fill_between(T_smooth, y_min, y_max, color=colors_fits[2], alpha=.2)
                        
                        df_sceg.loc[feat, 'RMS_DPL_'+data.index[0][:-4]] = np.sqrt(np.sum((DPL(T_conditions, *p_DPL) - data)**2))
                        df_sceg.loc[feat, 'condition_DPL_'+data.index[0][:-4]] = np.linalg.cond(cov)
                        
                    except: print('could not solve DPL for {}'.format(feat))    
                    if i_solve == len(p0s)-1: solving=False # you didn't solve it, but it's time to move on
                
                # housekeeping to make the plots look nice
                axs[i_1,i_2].set_ylabel(name_plot[i_plot])
                
                # DPL makes axis zoom out too much. specify zoom. 
                y_min = min(data-uc_data.to_numpy(float))
                y_max = max(data+uc_data.to_numpy(float))
                axs[i_1,i_2].set_ylim(y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
                
                axs[i_1,i_2].legend()
                
                
                for i_1 in range(np.shape(axs)[0]): 
                    for i_2 in range(np.shape(axs)[1]):
                        axs[i_1,i_2].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
                        axs[i_1,i_2].xaxis.set_minor_locator(AutoMinorLocator(5))
                        axs[i_1,i_2].yaxis.set_minor_locator(AutoMinorLocator(5))
                        
                        # set up x-axis for all plots
                        axs[-1,i_2].set_xlabel('Temperature (K)')
                        axs[-1,i_2].set_xticks(np.arange(300, 1301, 200))
                        axs[-1,i_2].set_xlim(200, 1400)
                
        plt.tight_layout()

        # if we plotted something, save the plot
        if (((~np.any(df_feat[pure_T_index]==-1))&(max(df_uc_sd_self)<0.1)) or
            ((~np.any(df_feat[air_T_index]==-1))&(max(df_uc_sd_air)<0.1))):
        
            if df_feat.doublets != []: plt.savefig(os.path.abspath('')+r'\plots\DPL\doublet {}.png'.format(feat), bbox_inches='tight',pad_inches = 0.1)
            else: plt.savefig(os.path.abspath('')+r'\plots\DPL\{}.png'.format(feat), bbox_inches='tight',pad_inches = 0.1)
    
            
        plt.close()
                 

#%% plot RMS data 




x_off = 10
y_off = 100

quanta = df_sceg.quanta.to_numpy(str)
quanta = [x.split()[0:6] for x in quanta]
y_offset = [(float(v1p) + float(v2p))/y_off for v1p,v2p,v3p, v1pp,v2pp,v3pp in quanta]
x_offset = [(float(v3p) + float(v2p))/x_off for v1p,v2p,v3p, v1pp,v2pp,v3pp in quanta]

plot_x = df_sceg.Jpp + x_offset
plot_y = df_sceg.Kcpp / df_sceg.Jpp + x_offset

names_base = ['RMS_HT_','RMS_labfit_','RMS_SPL_','RMS_DPL_']

# RMS for all values
for i_which, d_which in enumerate(d_whiches): 
    
    fig, axs = plt.subplots(3,4, figsize=(18,10), sharey = 'row', sharex = 'col', gridspec_kw={'hspace':0.03, 'wspace':0.01, 'width_ratios': [5,5,5,6]}) # sharex = 'col', sharey = 'row', 

    for i_param, param in enumerate(extra_params_base): 
        
        names_base_param = [x+param+'_'+d_which for x in names_base]
        plot_cmax = np.nanmax(df_sceg[names_base_param])
        plot_cmin = np.nanmin(df_sceg[names_base_param])
        
        for i_base, which_base in enumerate(names_base):
            
            if np.any(df_sceg[names_base_param[i_base]]): 
                
                plot_c = df_sceg[names_base_param[i_base]]
                
                sp1 = axs[i_param,i_base].scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', 
                                              norm=colors.LogNorm(vmin=plot_cmin, vmax=plot_cmax), zorder=2, linewidth=2)
                
                axs[i_param,i_base].text(7.5, 0.1, 'RMS = {:.3f}±{:.3f}'.format(np.nanmean(plot_c),np.nanstd(plot_c)))
                
            else: 
                
                axs[i_param,i_base].text(6.5, 0.1, 'No Data for Comparison')
        
        fig.colorbar(sp1, shrink=0.9, label='Width - RMS meas vs fit',  pad=0.01)

    axs[0,0].set_title('HITRAN')
    axs[0,1].set_title('SPL (Labfit)')
    axs[0,2].set_title('SPL (Scipy)')
    axs[0,3].set_title('DPL (Scipy)')
    
    axs[0,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    axs[1,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    axs[2,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    
    axs[2,0].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    axs[2,1].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    axs[2,2].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    
    
    
    for i_axs in [0,1,2]: 
        for j_axs in [0,1,2]: 
            axs[i_axs,j_axs].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
            axs[i_axs,j_axs].xaxis.set_minor_locator(AutoMinorLocator(5))
            axs[i_axs,j_axs].yaxis.set_minor_locator(AutoMinorLocator(5))
    

    plt.savefig(os.path.abspath('')+r'\plots\DPL\RMS {}.png'.format(d_which), bbox_inches='tight',pad_inches = 0.1)





#%% plot condition data (are we over fitting)


names_base = ['condition_SPL_','condition_DPL_']

fig, axs = plt.subplots(3,4, figsize=(18,10), sharey = 'row', sharex = 'col', gridspec_kw={'hspace':0.03, 'wspace':0.01, 'width_ratios': [5,6,5,6]}) # sharex = 'col', sharey = 'row', 


for i_which, d_which in enumerate(d_whiches): 
    
    for i_param, param in enumerate(extra_params_base): 
        
        names_base_param = [x+param+'_'+d_which for x in names_base]
        plot_cmax = np.nanmax(df_sceg[names_base_param][df_sceg[names_base_param] != np.inf])
        plot_cmin = np.nanmin(df_sceg[names_base_param])
        
        for i_base, which_base in enumerate(names_base):
            
            if np.any(df_sceg[names_base_param[i_base]]): 
                
                plot_c = df_sceg[names_base_param[i_base]]
                
                sp1 = axs[i_param,i_base].scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', 
                                              norm=colors.LogNorm(vmin=plot_cmin, vmax=plot_cmax), zorder=2, linewidth=2)
                
                axs[i_param,i_base+2*i_which].text(7.5, 0.1, 'RMS = {:.3f}±{:.3f}'.format(np.nanmean(plot_c),np.nanstd(plot_c)))
        
        if i_base+2*i_which == 3: 
            fig.colorbar(sp1, shrink=0.9, label='Width - RMS meas vs fit',  pad=0.01)

    axs[0,0].set_title('SPL (H$_{2}$O-H$_{2}$O)')
    axs[0,1].set_title('DPL (H$_{2}$O-H$_{2}$O)')
    axs[0,2].set_title('SPL (H$_{2}$O-air)')
    axs[0,3].set_title('DPL (H$_{2}$O-air)')
    
    axs[0,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    axs[1,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    axs[2,0].set_ylabel('Kc" / J" + (ν$_{1}$+ν$_{2}$)/100')
    
    axs[2,0].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    axs[2,1].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    axs[2,2].set_xlabel('J" + (ν$_{2}$+ν$_{3}$)/10')
    
    
    
    for i_axs in [0,1,2]: 
        for j_axs in [0,1,2]: 
            axs[i_axs,j_axs].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
            axs[i_axs,j_axs].xaxis.set_minor_locator(AutoMinorLocator(5))
            axs[i_axs,j_axs].yaxis.set_minor_locator(AutoMinorLocator(5))
    


plt.savefig(os.path.abspath('')+r'\plots\DPL\z - condition.png')








#%% plot RMS data as histograms




fig, axs = plt.subplots(3,3, figsize=(10,10), sharey = 'row', sharex = 'col', gridspec_kw={'hspace':0.01, 'wspace':0.03, 'width_ratios': [5,5,5]}) # sharex = 'col', sharey = 'row', 



bins = 115
ylim = 29

xlim_low = -0.002
xlim_high = 0.095

# width RMS values
axs[0,0].hist(RMS[:,3], bins=bins)
axs[1,0].hist(RMS[:,4], bins=bins)
axs[2,0].hist(RMS[:,5], bins=bins)

axs[0,0].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,3]))))
axs[1,0].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,4]))))
axs[2,0].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,5]))))

axs[2,0].set_xlim((xlim_low,xlim_high))
axs[2,0].set_xlabel('RMS - Collisional Width')

# shift RMS values
axs[0,1].hist(RMS[:,6], bins=bins)
axs[1,1].hist(RMS[:,7], bins=bins)
axs[2,1].hist(RMS[:,8], bins=bins)

axs[0,1].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,6]))))
axs[1,1].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,7]))))
axs[2,1].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,8]))))

axs[2,1].set_xlim((xlim_low,xlim_high))
axs[2,1].set_xlabel('RMS - Collisional Shift')

# SD RMS values
axs[0,2].hist(RMS[:,9], bins=bins)
axs[1,2].hist(RMS[:,10], bins=bins)
axs[2,2].hist(RMS[:,11], bins=bins)

axs[0,2].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,9]))))
axs[1,2].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,10]))))
axs[2,2].text(0.05, 26, 'Count = {}'.format(np.count_nonzero(~np.isnan(RMS[:,11]))))

axs[2,2].set_xlim((xlim_low,xlim_high))
axs[2,2].set_xlabel('RMS - Speed Dependence')



axs[0,0].set_ylim((0,ylim))
axs[1,0].set_ylim((0,ylim))
axs[2,0].set_ylim((0,ylim))

axs[0,0].set_ylabel('SPL (Labfit)')
axs[1,0].set_ylabel('SPL (Scipy)')
axs[2,0].set_ylabel('DPL (Scipy)')




for i_axs in [0,1,2]: 
    for j_axs in [0,1,2]: 
        axs[i_axs,j_axs].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs[i_axs,j_axs].xaxis.set_minor_locator(AutoMinorLocator(5))
        axs[i_axs,j_axs].yaxis.set_minor_locator(AutoMinorLocator(5))




plt.savefig(os.path.abspath('')+r'\plots\DPL\RMS - histogram.png')




