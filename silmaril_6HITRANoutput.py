r'''

silmaril6 - HITRAN output

processes labfit data for output into a HITRAN file format and for processing in silmaril 7


r'''



import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')


import linelist_conversions as db
import fig_format
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import labfithelp as lab

import os
import pickle
import pldspectrapy as pld
import td_support as td


# %% define some dictionaries and parameters

d_type = 'pure' # 'pure' or 'air'

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
props['gamma_air'] = ['gamma_air', 'γ air', 3, 25, 0.10] # ? (might be too generous for gamma only fits)
props['elower'] = ['elower', 'E\"', 4, 34, 200] # only float this when things are weird
props['n_air'] = ['n_air', 'n air', 5, 26, 0.13]
props['delta_air'] = ['delta_air', 'δ air', 6, 27, .005]
props['n_delta_air'] = ['n_delta_air', 'n δ air', 7, 28, 0.2]
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

d_labfit_kp2 = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool 2'
d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

base_name_pure = 'p2020'
d_cutoff_locations = d_labfit_main + '\\cutoff locations pure.pckl'

base_name_air = 'B2020Ja1'


n_update_name = 'n_gam'
if d_type == 'pure': base_name = base_name_pure + n_update_name
elif d_type == 'air': base_name = base_name_air + n_update_name

ratio_min_plot = -2 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
offset = 5 # for plotting
plot_spectra = False
df_calcs_dict = {}

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

cutoff_s296 = 1E-24 

d_sceg_load = r'D:\OneDrive - UCB-O365\water database\done'
bins_done = sorted(os.listdir(d_sceg_load), key=bin_names.index)

d_paul = r'C:\Users\scott\Documents\1-WorkStuff\Labfit\working folder\paul nate og\PaulData_SD_Avgn_AKn2'
d_sceg_save = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

d_HT2020 = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2020\H2O.par'
d_HT2016 = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2016\H2O.par'




please = stopherebeforeyougettoofar

# %% bin combiner - compile to export

d_sceg_bin = os.path.join(d_sceg_load, bins_done[0]) # where the saved file is located
d_sceg_bin_og = os.path.join(d_sceg_bin, os.listdir(d_sceg_bin)[1][:-4])

df_HT2020 = db.labfit_to_df(d_sceg_bin_og, htp=False) # open og (HITRAN) database - faster to do it all in the beginning

df_HT2020_HT = db.par_to_df(d_HT2020)
df_HT2020_HT.index = df_HT2020_HT.index +1

df_HT2016_HT = db.par_to_df(d_HT2016)
df_HT2016_HT.index = df_HT2016_HT.index +1


df_paul = db.labfit_to_df(d_paul, htp=False) # open paul database



for bin_name in bin_names: 
    if bin_name in bins_done: # sorts them according to bin_names (lazy but effective)
                
        d_saved = os.path.join(d_sceg_load, bin_name) # where the saved file is located

        [_, use_which_file] = lab.newest_rei(d_saved, bin_name)
            
        d_load = os.path.join(d_saved, use_which_file[:-4]) 
        d_load_og = os.path.join(d_saved, os.listdir(d_saved)[1][:-4]) 
        wvn_range = bins[bin_name][1:3]

        print('name:{}     load:{}     og:{}'.format(bin_name, d_load, d_load_og))

        df_bin = lab.trim(db.labfit_to_df(d_load, htp=False), wvn_range) # open and trim old database

        [T, P, wvn, trans, res, _, _, _] = lab.labfit_to_spectra(False, bins, bin_name, d_load=d_load) # get the spectra
        [_, _,   _,  _, res_og, _, _, _] = lab.labfit_to_spectra(False, bins, bin_name, d_load=d_load_og) # get the original spectra


        if plot_spectra: 
            
            if d_load[-3:] != '-og':
            
                df_calcs = lab.information_df('', bin_name, bins, cutoff_s296, T, df_external_load=df_bin) # <-------------------
                lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props['delta_self'], axis_labels=False) # <-------------------
                plt.title(bin_name)
                
                df_calcs_dict[bin_name] = df_calcs.copy()

        if bin_name == bins_done[0]: # if this is the first one
            
            if d_type == 'air': pass

            else:
                
                df_sceg_pure = df_bin.copy()
                
                T_pure = T.copy()
                P_pure = P.copy()
                wvn_pure = wvn.copy()
                trans_pure = trans.copy()
                res_pure = res.copy()
                res_og_pure = res_og.copy()


        else: # add on to existing dataframe

            if d_type == 'air': pass

            else: 
                
                df_sceg_pure = df_sceg_pure.append(df_bin)
                
                T_pure.extend(T)
                P_pure.extend(P)
                wvn_pure.extend(wvn)
                trans_pure.extend(trans)
                res_pure.extend(res)
                res_og_pure.extend(res_og)
                



please =sdfsdfssdf



# %% re-write quantum assignments in a way that is useful before saving

df_sceg = lab.information_df(False, False, False, cutoff_s296, T_pure, df_external_load=df_sceg_pure)

f = open(os.path.join(d_sceg_save,'df_sceg.pckl'), 'wb')
# pickle.dump([df_sceg, df_HT2020, df_paul], f)
pickle.dump([df_sceg, df_HT2020, df_HT2020_HT, df_HT2016_HT, df_paul], f)
f.close()

# f = open(os.path.join(d_sceg,'spectra_air.pckl'), 'wb')
# pickle.dump([T_air, P_air, wvn_air, trans_air, res_air, res_og_air], f)
# f.close()

f = open(os.path.join(d_sceg_save,'spectra_pure.pckl'), 'wb')
pickle.dump([T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure], f)
f.close()

please = stopfsdsaasd

# %% build new labfit INP file that has updates from pure water (for air-water)

lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file

# load in updated INP file (with the right measurements)
inp_main = open(os.path.join(d_labfit_main, 'p2020a.inp'), "r").readlines()

num_ASC = int(inp_main[0].split()[2])
# isolate part of INP without features (ASC files only)
inp_final = inp_main[:num_ASC*lines_per_asc + 3] 
# list of features (index+1 = labfit index)
features_list = np.array([float(i.split()[3]) for i in inp_main[num_ASC*lines_per_asc + 3:][0::4]])

i_stop_last = 0

# snag updated features from pure water measurements
for bin_name in bin_names: 
    if bin_name in bins_done: # sorts them according to bin_names (lazy but effective)
    
        d_saved = os.path.join(d_sceg_load, bin_name) # where the saved file is located

        [_, use_which_file] = lab.newest_rei(d_saved, bin_name)
            
        rei_load = open(os.path.join(d_saved, use_which_file[:-4]+'.rei') , "r").readlines()
        
        wvn_range = bins[bin_name][1:3]
        
        i_start = np.argmin(abs(wvn_range[0] - features_list))
        if bin_name == bins_done[0]: # if this is the first one
            i_start = 1 # keep the first features
        
        i_stop = np.argmin(abs(wvn_range[1] - features_list))-1 # take off one to avoid overlap with next bin
        if bin_name == bins_done[-1]: # if this is the last one
            i_stop = 43477 # keep the laser features

        num_ASC_bin = int(rei_load[0].split()[2])
       
        # isolate part of INP without features (ASC files only)
        inp_features_bin = rei_load[num_ASC_bin*lines_per_asc + 3:]
        
        i_stop_actual = i_stop
        
        while int(inp_features_bin[i_stop_actual*4-4].split()[0]) != i_stop:
            
            i_stop_actual += 1
            
            if i_stop_actual - i_stop > 50: time = tostopplease # no bin has that many new features, you missed your target
                
        inp_features_bin = inp_features_bin[i_start*4-4:i_stop_actual*4]
        
        if int(inp_features_bin[0].split()[0]) != i_start: messed = up_start
        if int(inp_features_bin[-4].split()[0]) != i_stop: messed = up_stop
        
        
        if bin_name == bins_done[0]: # if this is the first one
        
            inp_features = inp_features_bin.copy()
        
        else: 
        
            inp_features.extend(inp_features_bin)
            
        print('name:{}     start:{}     stop:{}    delta with last:{}'.format(bin_name, i_start, i_stop, i_start-i_stop_last))
        
        i_stop_last = i_stop
        

inp_features_2 = inp_features.copy()

# walk through feature by feature to make some changes  
for i in range(0, len(inp_features)//4+1):
    
    print(i)
    
    # reset all floats to 1 (no float)
    inp_features[4*i+2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n' 

    # set all SD to 0.12
    inp_features[4*i+1] = inp_features[4*i+1][:65] + '  0.12000\n'
    
    # set all air shifts to 0 (linear -> exp.), shift exponenets to 1.0
    inp_features[4*i] = inp_features[4*i][:72] + '  0.0000000  1.00000000' + inp_features[4*i][95:]

inp_final.extend(inp_features)

inp_final[0] = inp_final[0][:33] + str(len(inp_features)//4) + inp_final[0][38:]


open(os.path.join(d_labfit_main, 'p2020a_updated.inp'), 'w').writelines(inp_final)




#%% export to par file - UPDATE TO INCLUDE THE NON-HITRAN PARAMETERS

f = open(os.path.join(d_sceg,'df_sceg.pckl'), 'rb')
[df_sceg, df_HT2020, df_HT2020_HT, df_HT2016_HT, df_paul] = pickle.load(f)
f.close()

par_name = 'H2O'

df_sceg2 = df_sceg.rename(columns={"n_delta_self": "delta_self", "uc_n_delta_self": "uc_deltap_self", 
                                   "n_delta_air": "delta_air", "uc_n_delta_air": "uc_deltap_air"})  # if using linear pressure shift model

db.df_to_par(df_sceg2, par_name, extra_params=db.SDVoigt_LinearShift, save_dir=d_sceg)

pld.db_begin('data - sceg')  # load the linelists into Python (keep out of the for loop)

mod, pars = td.spectra_single_lmfit()
wvn = np.arange(6660,7202,0.005) # wavenumber range

pars['mol_id'].value = 1 # water = 1 (hitran molecular code)
pars['pathlength'].value = 1 # pathlength in cm

pars['pressure'].value = 10/760 # pressure in atm (converted from Torr)
pars['molefraction'].value = 1 # mole fraction

pars['temperature'].value = 300 # temperature in K
model_TD = mod.eval(xx=wvn, params=pars, name='H2O')
model_trans300 = np.exp(-np.real(np.fft.rfft(model_TD)))

pars['temperature'].value = 1000 # temperature in K
model_TD = mod.eval(xx=wvn, params=pars, name='H2O')
model_trans1000 = np.exp(-np.real(np.fft.rfft(model_TD)))

pars['temperature'].value = 2000 # temperature in K
model_TD = mod.eval(xx=wvn, params=pars, name='H2O')
model_trans2000 = np.exp(-np.real(np.fft.rfft(model_TD)))

plt.plot(wvn, model_trans300, label='300')
plt.plot(wvn, model_trans1000, label='1000')
plt.plot(wvn, model_trans2000, label='2000')
plt.legend()
