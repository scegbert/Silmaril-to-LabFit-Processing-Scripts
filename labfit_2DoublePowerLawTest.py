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

# pure water data
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
                           'quanta','local_iso_id','vp','vpp','Jp','Kap','Kcp','Jpp','Kapp','Kcpp','m','doublets']]


df_sceg = pd.merge(df_sceg_pure, df_sceg_air, on='index')


#%% prepare transitions we will use to explore DPL


bin_names_test = ['B10', 
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
                  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
                  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47']

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
                    [11787, 11788, 12001, 12136],
                    [12647, 12837, 12922, 12923, 12952, 13048],
                    [13848, 13867, 13873, 13886, 13950, 14026, 14025],
                    [14045, 14060, 14169, 14172, 14459, 14509, 14740, 14742, 14817],
                    [15077, 15097, 15128, 15126, 15146, 15144, 15194, 15487, 15535, 15596],
                    [15812, 16182, 16210, 16209, 16259, 16295, 16467, 16551, 16558, 16572, 16594, 16596],
                    [16735, 17112, 17170, 17290, 17295, 17300, 17304, 17309, 17339, 17383, 17423, 17428, 17475, 17473],
                    [17677, 17856, 17955, 18011],
                    [18339, 18398, 18406, 18478, 18555, 18611, 18974, 19055],
                    [19207, 19281, 19333, 19339, 19346, 19398, 19406, 19421, 19422, 19463, 19707, 19814, 19812],
                    [20283, 20315, 20320, 20349, 20429, 20464, 20465, 20550, 20835],
                    [21075, 21189, 21361, 21430, 21455, 21459, 21484],
                    [21873, 21929, 21991, 22025, 22060, 22337, 22422, 22431, 22455],
                    [22562, 22611, 22707, 22855, 22893, 23161, 23200, 23250, 23360, 23378],
                    [23499, 23615, 24128, 24166, 24167],
                    [24324, 24469, 24467, 24484, 24506, 24605, 24802, 24840, 24841],
                    [25117, 25118, 25176, 25210, 25233, 25341, 25340, 25495, 25502, 25534, 25695, 25732],
                    [25931, 25935, 26046, 26067, 26117, 26118, 26134, 26190, 26365, 26463, 26578, 26611],
                    [26691, 26745, 26750, 27034, 27107, 27181, 27194, 27227, 27244, 27268, 27282, 27307, 27334, 27348, 27442, 27475],
                    [27622, 27698, 27730, 27839, 28117, 28152, 28187],
                    [28492, 28543, 28805, 28824, 29054, 29069],
                    [29433, 29524, 29550, 29743, 29840, 29937, 30022, 30272],
                    [30697, 30781, 30851, 30999, 31006, 30610, 30612],
                    [31330, 31379, 31504, 31555, 31565, 31680, 31761, 32034],
                    [32116, 32145, 32215, 32308, 32453, 32506, 32634, 32767, 32805, 32870],
                    [32958, 33055, 33056, 33197, 33209, 33330, 33347, 33596, 33603, 33612],
                    [33706, 33757, 33786, 33801, 33811, 33956, 34111, 34228, 34245, 34243, 34360],
                    [34403, 34413, 34508, 34537, 34617, 34662, 34834, 34892, 34906, 34907, 34917, 34962, 35005, 35036],
                    [35195, 35251, 35348, 35597],
                    [35745, 35926, 36029],
                    [36383]]

features_doublets = [[],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [],
                     [[11787, 11788]],
                     [[12922, 12923]],
                     [[14026, 14025]],
                     [[14169, 14172]],
                     [[15128, 15126], [15146, 15144]],
                     [[16210, 16209], [16594, 16596]],
                     [[17304, 17309], [17475, 17473]],
                     [],
                     [],
                     [[19421, 19422], [19814, 19812]],
                     [[20464, 20465]],
                     [[21455, 21459]],
                     [],
                     [],
                     [[24166, 24167]],
                     [[24469, 24467], [24841, 24840]],
                     [[25118, 25117], [25340, 25341]],
                     [[25935, 25931], [26117, 26118]],
                     [[26750, 26745]],
                     [],
                     [],
                     [],
                     [[30610, 30612]],
                     [],
                     [],
                     [[33055, 33056]],
                     [[34245, 34243]],
                     [[34906, 34907]],
                     [],
                     [],
                     []]


features_strong_flat_all = [item for sublist in features_strong for item in sublist]
features_strong_flat = features_strong_flat_all.copy()

for doublet in features_doublets: 
    if doublet != []: 
        for sub_doublet in doublet: 
            features_strong_flat.remove(sub_doublet[1])

T_conditions = list(dict.fromkeys([i.split()[0] for i in d_conditions]))

df_sceg = df_sceg[df_sceg.index.isin(features_strong_flat_all)]



extra_params_base = ['gs','sds','ds','ga','sda','da'] # self and air width, SD, and shift
extra_params = ['nu_300', 'uc_nu_300']

for i_p, param in enumerate(extra_params_base): 
    for i_T, T in enumerate(T_conditions):
        
        name = param + '_' + T
        
        extra_params.append(name)
        extra_params.append('uc_' + name)
    
    

df_sceg.reindex(columns=extra_params)


#%% perform DPL calculations

d_labfit_kernal = d_labfit_kp1
d_old_all = r'E:\water database' # for comparing to original input files

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)


for i_bin, bin_name in enumerate(bin_names_test):

    features = features_strong[i_bin]
    features_constrain = features_doublets[i_bin]
            
    for i_type, d_type in enumerate(['pure', 'air']): 

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
            
            if feat in features_strong_flat: 
            
                i_feat = features_strong_flat.index(feat)
            
                output_lab[i_type,i_feat,0] = df_calcs.loc[feat].nu
                output_lab[i_type,i_feat,1] = df_calcs.loc[feat].uc_nu
                output_lab[i_type,i_feat,2] = df_calcs.loc[feat]['gamma_'+d_which]
                output_lab[i_type,i_feat,3] = df_calcs.loc[feat]['uc_gamma_'+d_which]
                output_lab[i_type,i_feat,4] = df_calcs.loc[feat]['n_'+d_which]
                output_lab[i_type,i_feat,5] = df_calcs.loc[feat]['uc_n_'+d_which]
                output_lab[i_type,i_feat,6] = df_calcs.loc[feat].sd_self
                output_lab[i_type,i_feat,7] = df_calcs.loc[feat].uc_sd_self
                output_lab[i_type,i_feat,8] = df_calcs.loc[feat]['delta_'+d_which]
                output_lab[i_type,i_feat,9] = df_calcs.loc[feat]['uc_delta_'+d_which]
                output_lab[i_type,i_feat,10] = df_calcs.loc[feat]['n_delta_'+d_which]
                output_lab[i_type,i_feat,11] = df_calcs.loc[feat]['uc_n_delta_'+d_which]
                
            
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
                        
            # update nu for all conditions that aren't this one
            if (T_iter == '300' and d_type == 'pure') is False:
                
                for feat in features: 
                    
                    if feat in features_strong_flat: 
                    
                        i_feat = features_strong_flat.index(feat) 
                        
                        # wavenumber as string
                        nu_str = '{:.7f}'.format(output_dpl[0,i_feat,0,6]) 
                        
                        # find the feature
                        i_guess = feat*lines_per_feature
                        
                        if int(inp_features[i_guess].split()[0]) != feat: 
                            i_guess = lab.floated_line_moved(i_guess+2, feat, inp_features, lines_per_feature)
                            i_guess -=2
                            
                        inp_features[i_guess] = inp_features[i_guess][:15] + nu_str + inp_features[i_guess][27:]
                        
                        # update width and shift to match values obtained for that conditions
                        if d_type == 'air': 
    
                            g_self_T = '{:.5f}'.format(output_dpl[0,i_feat,i_T,0])
                            d_self_T = '{:.7f}'.format(output_dpl[0,i_feat,i_T,4])
                            if d_self_T[0] != '-': d_self_T = ' '+d_self_T
                            
                            inp_features[i_guess+1] = '   ' + g_self_T + inp_features[i_guess+1][10:19] + d_self_T + inp_features[i_guess+1][29:]
                            
            inp_header[0] = inp_header[0][:25] + '    {}  '.format(iter_asc) + inp_header[0][32:]
            
            inp_updated = inp_header.copy()
            inp_updated.extend(inp_asc)
            inp_updated.extend(inp_features)
            
            open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
            
            print('\n************************            {}          {}        {}\n'.format(bin_name, d_type, T_iter))
                        
            # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
            if T_iter == '300' and d_type == 'pure': 
                lab.float_lines(d_labfit_kernal, bin_name, features, props['nu'], 'inp_new', features_constrain) 
            
            lab.float_lines(d_labfit_kernal, bin_name, features, props['gamma_'+d_which], 'inp_new', features_constrain) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['sd_self'], 'inp_new', features_constrain) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['delta_'+d_which], 'inp_new', []) 
            
            # run labfit
            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
            
            if feature_error is None: 
                
                # plot results (at least at first to make sure things aren't crazy)
                # [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
                df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                # lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>0], 2, props['gamma_'+d_which], props['nu'], features=features, axis_labels=False) # <-------------------
                # plt.title(bin_name + ' - ' + T_iter)
        
        
                # extract updated values and compile into dict/list
                for feat in features: 
                    
                    if feat in features_strong_flat: 
                    
                        i_feat = features_strong_flat.index(feat) 
                    
                        output_dpl[i_type,i_feat,i_T,0] = df_calcs.loc[feat]['gamma_'+d_which]
                        output_dpl[i_type,i_feat,i_T,1] = df_calcs.loc[feat]['uc_gamma_'+d_which]
                        
                        output_dpl[i_type,i_feat,i_T,2] = df_calcs.loc[feat].sd_self
                        output_dpl[i_type,i_feat,i_T,3] = df_calcs.loc[feat].uc_sd_self
                        output_dpl[i_type,i_feat,i_T,4] = df_calcs.loc[feat]['delta_'+d_which]
                        output_dpl[i_type,i_feat,i_T,5] = df_calcs.loc[feat]['uc_delta_'+d_which]
                        
                        output_dpl[i_type,i_feat,i_T,6] = df_calcs.loc[feat].nu
                        output_dpl[i_type,i_feat,i_T,7] = df_calcs.loc[feat].uc_nu
                        
            else:  
                 
                for feat in features: 
                                        
                    if feat in features_strong_flat: 
                    
                        i_feat = features_strong_flat.index(feat)
                        
                        features_iter = [feat].copy()
                        features_constrain_iter = []
                        for doublet in features_constrain: 
                            if feat in doublet: 
                                features_iter = doublet.copy()
                                features_constrain_iter = [doublet.copy()]
                        # features_constrain = Just constrain them all, but don't float them all
                        
                        open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
                        
                        print('\n            trying one at a time, currently on {}'.format(feat))                                    
                        
                        # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
                        if T_iter == '300' and d_type == 'pure': 
                            lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['nu'], 'inp_new', features_constrain_iter) 
                        
                        lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['gamma_'+d_which], 'inp_new', features_constrain_iter) 
                        lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['sd_self'], 'inp_new', features_constrain_iter) 
                        lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['delta_'+d_which], 'inp_new', []) 
                        
                        # run labfit
                        feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
    
                        if feature_error is not None: 
    
    
                            open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
                        
                            print('\n            trying without shift (but with nu), still taking things one at a time')
                                        
                            # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
                            lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['nu'], 'inp_new', features_constrain_iter) 
                            lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['gamma_'+d_which], 'inp_new', features_constrain_iter) 
                            lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['sd_self'], 'inp_new', features_constrain_iter)  
                            
                            # run labfit
                            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
                            
                            if feature_error is not None: 
                            
                                open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
                            
                                print('\n            trying without shift (but with nu) or SD, still taking things one at a time')
                                            
                                # float lines we're investigating (gamma, SD, delta), constrain all values for doublets            
                                lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['nu'], 'inp_new', features_constrain_iter) 
                                lab.float_lines(d_labfit_kernal, bin_name, features_iter, props['gamma_'+d_which], 'inp_new', features_constrain_iter) 
                                
                                # run labfit
                                feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
                                                    
                        if feature_error is None: 
                            
                            df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                            
                            output_dpl[i_type,i_feat,i_T,0] = df_calcs.loc[feat]['gamma_'+d_which]
                            output_dpl[i_type,i_feat,i_T,1] = df_calcs.loc[feat]['uc_gamma_'+d_which]
        
                            output_dpl[i_type,i_feat,i_T,2] = df_calcs.loc[feat].sd_self
                            output_dpl[i_type,i_feat,i_T,3] = df_calcs.loc[feat].uc_sd_self
                            output_dpl[i_type,i_feat,i_T,4] = df_calcs.loc[feat]['delta_'+d_which]
                            output_dpl[i_type,i_feat,i_T,5] = df_calcs.loc[feat]['uc_delta_'+d_which]
                            
                            output_dpl[i_type,i_feat,i_T,6] = df_calcs.loc[feat].nu
                            output_dpl[i_type,i_feat,i_T,7] = df_calcs.loc[feat].uc_nu




f = open(os.path.join(d_sceg_save,'DPL exploration.pckl'), 'wb')
pickle.dump([output_dpl, output_lab, T_conditions, features_strong, features_doublets], f)
f.close()               



#%% load in the DPL data


f = open(os.path.join(d_sceg_save,'DPL exploration.pckl'), 'rb')
[output_dpl, output_lab, T_conditions, features_strong, features_doublets] = pickle.load(f)
f.close()     

T_conditions = [float(T) for T in T_conditions]
T_conditions = np.asarray(T_conditions)

features_strong_flat = [item for sublist in features_strong for item in sublist] # list of all transitions, only one from each doublet included

for doublet in features_doublets: 
    if doublet != []: 
        for sub_doublet in doublet: 
            features_strong_flat.remove(sub_doublet[1])

df_sceg






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

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

def SPL(T, c, n): 
    
    return c*(296/T)**n

def SPLoff(T, c1, c2, n): 

    return c1*(296/T)**n + c2
    
def DPL(T, c1, n1, c2, n2): 
    
    return c1*(296/T)**n1 + c2*(296/T)**n2



please = stophere_fullyloaded

#%% plots for Bob

T_smooth = np.linspace(T_conditions[0], T_conditions[-1], num=1000)
df_quanta = db.labfit_to_df('E:\water database\pure water\B1\B1-000-HITRAN')

i_type = 1 # 0 = pure water, 1 = air-water absorption
i_RMS = False

features_doublets_flat = [item for sublist in features_doublets for item in sublist]
features_doublets_flat = [item for sublist in features_doublets_flat for item in sublist]

column_names=['quanta', 
              'g_300','uc_g_300','g_500','uc_g_500','g_700','uc_g_700','g_900','uc_g_900','g_1100','uc_g_1100','g_1300','uc_g_1300', 
              'sd_300','uc_sd_300','sd_500','uc_sd_500','sd_700','uc_sd_700','sd_900','uc_sd_900','sd_1100','uc_sd_1100','sd_1300','uc_sd_1300', 
              'd_300','uc_d_300','d_500','uc_d_500','d_700','uc_d_700','d_900','uc_d_900','d_1100','uc_d_1100','d_1300','uc_d_1300']

df_features = pd.DataFrame(columns=column_names)

list_feature = np.zeros_like(column_names)

for i_feat, feature in enumerate(features_strong_flat): 
    
    if feature not in features_doublets_flat: 
          
        # confirm all values are floated
        if np.all(output_dpl[:,i_feat,:,1] != -1) & np.all(output_dpl[:,i_feat,:,3] != -1) & (
            np.all(output_dpl[:,i_feat,:,5] != -1)) & np.all(output_dpl[:,i_feat,1,7] == -1): 
             
            for i_T, T in enumerate(T_conditions): 
                
                T_str = str(int(T))
                
                list_feature[i_T*2 + 0 + 1] = output_dpl[i_type,i_feat,i_T,0] 
                list_feature[i_T*2 + 1 + 1] = output_dpl[i_type,i_feat,i_T,1]

                list_feature[i_T*2 + 12 + 1] = output_dpl[i_type,i_feat,i_T,2]
                list_feature[i_T*2 + 13 + 1] = output_dpl[i_type,i_feat,i_T,3]

                list_feature[i_T*2 + 24 + 1] = output_dpl[i_type,i_feat,i_T,4]
                list_feature[i_T*2 + 25 + 1] = output_dpl[i_type,i_feat,i_T,5]
                
            df_features.loc[feature] = list_feature
            
            df_features.quanta[feature] = df_quanta.quanta[feature]
            
            
            fig, axs = plt.subplots(4,2, figsize=(14,25), sharex = 'col', gridspec_kw={'hspace':0}) # sharex = 'col', sharey = 'row', 
            
            fig.subplots_adjust(top=0.95)
            
           
            quanta = df_quanta.quanta[feature].split()
            try: kcJ = float(quanta[8])/float(quanta[6])
            except: kcJ = 0.
            
            plt.suptitle('{}{}{} ← {}{}{}      {}$_{}$$_{}$ ← {}$_{}$$_{}$      Kc" / J" = {:.3f}'.format(quanta[0],quanta[1],quanta[2],quanta[3],quanta[4],quanta[5],
                                                                            quanta[6],quanta[7],quanta[8],quanta[9],quanta[10],quanta[11], kcJ))
            RMS_params = 9
            RMS_extra = 3

            RMS_iter = np.empty((1,RMS_extra + RMS_params*2))
            RMS_iter[:] = np.nan
                        
            RMS_iter[0,0] = feature
            RMS_iter[0,1] = float(quanta[6]) # J"
            RMS_iter[0,2] = kcJ # Kc / J"
            
            
            for i_type, name_type in enumerate(['H2O-H2O','air-H2O ']): 
                
                # --------------------- collisional widths           
                # fit the data           
                y_data = output_dpl[i_type,i_feat,:,0]
                y_unc = output_dpl[i_type,i_feat,:,1]
                
                if i_type == 0: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0027)**2 + (0.0081)**2) * y_data
                elif i_type == 1: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0025)**2 + (0.029)**2 + (0.0082)**2) * y_data
                
                p_labfit = [output_lab[i_type,i_feat,2], output_lab[i_type,i_feat,4]]
    
                # plot the data
                axs[0,i_type].plot(T_conditions, y_data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                axs[0,i_type].errorbar(T_conditions, y_data, y_unc, color='k', fmt='none', capsize=5, zorder=10)
                
                axs[0,i_type].plot(T_smooth, SPL(T_smooth, *p_labfit), label='SPL (Labfit)') 
                RMS_iter[0,RMS_extra + 0 + i_type*RMS_params] = np.sqrt(np.mean((SPL(T_conditions, *p_labfit) - y_data)**2))
                
                try: 
                    p_SPL, _ = fit(SPL, T_conditions, y_data, p0=p_labfit, maxfev=5000)
                    axs[0,i_type].plot(T_smooth, SPL(T_smooth, *p_SPL), label='SPL (Scipy)') 
                    RMS_iter[0,RMS_extra + 1 + i_type*RMS_params] = np.sqrt(np.mean((SPL(T_conditions, *p_SPL) - y_data)**2))
                except: pass
            
                try: 
                    p_DPL, _ = fit(DPL, T_conditions, y_data, p0=p_labfit + [x/10 for x in p_labfit], maxfev=5000)
                    axs[0,i_type].plot(T_smooth, DPL(T_smooth, *p_DPL), label='DPL (Scipy)') 
                    RMS_iter[0,RMS_extra + 2 + i_type*RMS_params] = np.sqrt(np.mean((DPL(T_conditions, *p_DPL) - y_data)**2))
                except: pass
                
                axs[0,i_type].set_ylabel(name_type + ' Width')
    
                axs[0,i_type].legend(loc='upper right')
    
                
                
                # --------------------- collisional shifts       
                # fit the data           
                y_data = output_dpl[i_type,i_feat,:,4]
                y_unc = output_dpl[i_type,i_feat,:,5]
                
                
                if i_type == 0: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0027)**2 + (0.0081)**2 + (1.7E-4 / (0.021*y_data))**2) * abs(y_data)
                elif i_type == 1: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0025)**2 + (0.029)**2 + (0.0082)**2 + (1.7E-4 / (0.789*y_data))**2) * abs(y_data)
                
                
                p_labfit = [output_lab[i_type,i_feat,8], output_lab[i_type,i_feat,10]]
    
                # plot collisional shifts
                axs[1,i_type].plot(T_conditions, y_data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                axs[1,i_type].errorbar(T_conditions, y_data, y_unc, color='k', fmt='none', capsize=5, zorder=10)
                
                delta_nu = output_lab[i_type,i_feat,0] - output_dpl[i_type,i_feat,i_T,6]
                
                axs[1,i_type].plot(T_smooth, SPL(T_smooth, *p_labfit), label='SPL (Labfit)') 
                RMS_iter[0,RMS_extra + 3 + i_type*RMS_params] = np.sqrt(np.mean((SPL(T_conditions, *p_labfit) - y_data)**2))
                
                try: 
                    p_SPL, _ = fit(SPL, T_conditions, y_data, p0=p_labfit, maxfev=5000)
                    axs[1,i_type].plot(T_smooth, SPL(T_smooth, *p_SPL), label='SPL (Scipy)') 
                    RMS_iter[0,RMS_extra + 4 + i_type*RMS_params] = np.sqrt(np.mean((SPL(T_conditions, *p_labfit) - y_data)**2))
                except: pass
            
                try: 
                    p_DPL, _ = fit(DPL, T_conditions, y_data, p0=p_labfit + [x/10 for x in p_labfit], maxfev=5000)
                    axs[1,i_type].plot(T_smooth, DPL(T_smooth, *p_DPL), label='DPL (Scipy)') 
                    RMS_iter[0,RMS_extra + 5 + i_type*RMS_params] = np.sqrt(np.mean((DPL(T_conditions, *p_DPL) - y_data)**2))
                except: pass
    
    
                axs[1,i_type].set_ylabel(name_type + ' Shift')
    
                axs[1,i_type].legend()
                
              
                
                # --------------------- aw (SD)
                
                y_data = output_dpl[i_type,i_feat,:,2]*output_dpl[i_type,i_feat,:,0]
                y_unc = output_dpl[i_type,i_feat,:,1] # estimate - using width uncertianty as stand-in
                
                if i_type == 0: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0027)**2 + (0.0081)**2 + 0.039**2) * y_data
                elif i_type == 1: y_unc = np.sqrt((y_unc/y_data)**2 + (0.0025)**2 + (0.029)**2 + (0.0082)**2 + 0.052**2) * y_data               
                
                
                # plot the data
                axs[2,i_type].plot(T_conditions, output_dpl[i_type,i_feat,:,2], color='k', marker='x', markersize=10, markeredgewidth=3, 
                            linestyle='None', label='Measurement', zorder=10)
                axs[2,i_type].errorbar(T_conditions, output_dpl[i_type,i_feat,:,2], output_dpl[i_type,i_feat,:,3], 
                                color='k', fmt='none', capsize=5, zorder=10)
            
                axs[2,i_type].plot(T_smooth, output_lab[i_type,i_feat,6]*np.ones_like(T_smooth), label='Average (Labfit)')          
    
                axs[2,i_type].set_ylabel(name_type + ' SD aw')
    
                axs[2,i_type].legend()
    
                
                
                # --------------------- gamma_SD
                # fit the data           
                y_data = output_dpl[i_type,i_feat,:,2]*output_dpl[i_type,i_feat,:,0]
                y_unc = output_dpl[i_type,i_feat,:,1] # estimate - using width uncertianty as stand-in
                
                p_labfit = [output_lab[i_type,i_feat,2], output_lab[i_type,i_feat,4]] # estimate - using width SPL as stand-in
    
                # plot the data
                axs[3,i_type].plot(T_conditions, y_data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                axs[3,i_type].errorbar(T_conditions, y_data, y_unc, color='k', fmt='none', capsize=5, zorder=10)
                
                axs[3,i_type].plot(T_smooth, output_lab[i_type,i_feat,6] * SPL(T_smooth, *p_labfit), label='SPL (Labfit - width)')
                RMS_iter[0,RMS_extra + 6 + i_type*RMS_params] = np.sqrt(np.mean((output_lab[i_type,i_feat,6] * SPL(T_conditions, *p_labfit) - y_data)**2))
                
                try: 
                    p_SPL, _ = fit(SPL, T_conditions, y_data, p0=p_labfit, maxfev=5000)
                    axs[3,i_type].plot(T_smooth, SPL(T_smooth, *p_SPL), label='SPL (Scipy)')
                    RMS_iter[0,RMS_extra + 7 + i_type*RMS_params] = np.sqrt(np.mean((output_lab[i_type,i_feat,6] * SPL(T_conditions, *p_SPL) - y_data)**2))
                except: pass
            
                try: 
                    p_DPL, _ = fit(DPL, T_conditions, y_data, p0=p_labfit + [x/10 for x in p_labfit], maxfev=5000)
                    axs[3,i_type].plot(T_smooth, DPL(T_smooth, *p_DPL), label='DPL (Scipy)') 
                    RMS_iter[0,RMS_extra + 8 + i_type*RMS_params] = np.sqrt(np.mean((output_lab[i_type,i_feat,6] * DPL(T_conditions, *p_DPL) - y_data)**2))
                except: pass
    
                
                axs[3,i_type].set_ylabel(name_type + ' SD Width')
    
                axs[3,i_type].legend(loc='upper right')
                
                # set up x-axis for all plots
                axs[-1,i_type].set_xlabel('Temperature (K)')
                axs[-1,i_type].set_xticks(np.arange(300, 1301, 200))
                axs[-1,i_type].set_xlim(200, 1400)
                
                
                for i_axs in range(len(axs)): 
                    axs[i_axs,i_type].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
                    axs[i_axs,i_type].xaxis.set_minor_locator(AutoMinorLocator(5))
                    axs[i_axs,i_type].yaxis.set_minor_locator(AutoMinorLocator(5))
                
            
            # plt.savefig(os.path.abspath('')+r'\plots\DPL\{}.png'.format(feature))
            plt.close()
                     
            if i_RMS: 
                RMS = np.vstack((RMS.copy(), RMS_iter.copy()))
                quanta_all.append(quanta[0:6].copy())
            else: 
                RMS = RMS_iter.copy(); i_RMS = True
                quanta_all = [quanta[0:6].copy()]
            
            
# df_features.to_csv('DPL_parameters.csv')


#%% plot RMS data 


fig, axs = plt.subplots(3,3, figsize=(10,10), sharey = 'row', sharex = 'col', gridspec_kw={'hspace':0.03, 'wspace':0.01, 'width_ratios': [5,5,6]}) # sharex = 'col', sharey = 'row', 


x_off = 10
y_off = 100

y_offset = [(float(v1p) + float(v2p))/y_off for v1p,v2p,v3p, v1pp,v2pp,v3pp in quanta_all]
x_offset = [(float(v3p) + float(v2p))/x_off for v1p,v2p,v3p, v1pp,v2pp,v3pp in quanta_all]

# width RMS values
axs[0,0].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,3], cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=np.nanmax(RMS[:,3:5+1]))
axs[0,1].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,4], cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=np.nanmax(RMS[:,3:5+1]))
sp1 = axs[0,2].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,5], cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=np.nanmax(RMS[:,3:5+1]))

fig.colorbar(sp1, shrink=0.9, label='Width - RMS meas vs fit',  pad=0.01)



# shift RMS values
axs[1,0].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,6], cmap='viridis', zorder=2, linewidth=2) #, vmin=0, vmax=18)
axs[1,1].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,7], cmap='viridis', zorder=2, linewidth=2)
sp2 = axs[1,2].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,8], cmap='viridis', zorder=2, linewidth=2)

fig.colorbar(sp2, shrink=0.9, label='Shift - RMS meas vs fit',  pad=0.01)

# SD RMS values
axs[2,0].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,9], cmap='viridis', zorder=2, linewidth=2) #, vmin=0, vmax=18)
axs[2,1].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,10], cmap='viridis', zorder=2, linewidth=2)
sp3 = axs[2,2].scatter(RMS[:,1]+x_offset, RMS[:,2]+y_offset, marker='x', c=RMS[:,11], cmap='viridis', zorder=2, linewidth=2)

fig.colorbar(sp3, shrink=0.9, label='SD - RMS meas vs fit',  pad=0.01)


axs[0,0].set_title('SPL (Labfit)')
axs[0,1].set_title('SPL (Scipy)')
axs[0,2].set_title('DPL (Scipy)')

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

        axs[i_axs,j_axs].text(4.5, 0, 'RMS = {:.3f}±{:.3f}'.format(np.nanmean(RMS[:,3+i_axs+3*j_axs]),np.nanstd(RMS[:,3+i_axs+3*j_axs])))

        # axs[i_axs,j_axs].scatter(RMS[:,1], RMS[:,2], marker='.', color='r', s=1, zorder=10)



plt.savefig(os.path.abspath('')+r'\plots\DPL\RMS.png')



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




