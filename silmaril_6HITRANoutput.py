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

if d_type == 'pure': d_sceg_load = r'H:\water database\pure water'
elif d_type == 'air': d_sceg_load = r'H:\water database\air water'
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


df_paul = db.labfit_to_df(d_paul, htp=False) # open paul databaseI



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
            
            df_sceg_all = df_bin.copy()
            
            T_all = T.copy()
            P_all = P.copy()
            wvn_all = wvn.copy()
            trans_all = trans.copy()
            res_all = res.copy()
            res_og_all = res_og.copy()


        else: # add on to existing dataframe

            
            df_sceg_all = df_sceg_all.append(df_bin)
            
            T_all.extend(T)
            P_all.extend(P)
            wvn_all.extend(wvn)
            trans_all.extend(trans)
            res_all.extend(res)
            res_og_all.extend(res_og)
            



# please =sdfsdfssdf



# %% save data

df_sceg = lab.information_df(False, False, False, cutoff_s296, T_all, df_external_load=df_sceg_all)

if d_type == 'pure': f = open(os.path.join(d_sceg_save,'df_sceg_pure.pckl'), 'wb')
elif d_type == 'air': f = open(os.path.join(d_sceg_save,'df_sceg_air.pckl'), 'wb')
pickle.dump([df_sceg, df_HT2020, df_HT2020_HT, df_HT2016_HT, df_paul], f)
f.close()

if d_type == 'pure': f = open(os.path.join(d_sceg_save,'spectra_pure.pckl'), 'wb')
elif d_type == 'air': f = open(os.path.join(d_sceg_save,'spectra_air.pckl'), 'wb')
pickle.dump([T_all, P_all, wvn_all, trans_all, res_all, res_og_all], f)
f.close()

# please = stopfsdsaasd


# %% load both databases and combine

# load pure water database
f = open(os.path.join(d_sceg_save,'df_sceg_pure.pckl'), 'rb')
[df_sceg_pure, _, _, _, _] = pickle.load(f)
f.close()

# load pure water database
f = open(os.path.join(d_sceg_save,'df_sceg_air.pckl'), 'rb')
[df_sceg_air, _, _, _, _] = pickle.load(f)
f.close()

# rename SD in air data to SD_air
df_sceg_air = df_sceg_air.rename(columns={"sd_self": "sd_air", "uc_sd_self": "uc_sd_air"}) 

# remove things we don't need from air data
df_sceg_air2 = df_sceg_air[['gamma_air', 'uc_gamma_air', 
                            'n_air', 'uc_n_air',
                            'sd_air', 'uc_sd_air',
                            'delta_air', 'uc_delta_air',
                            'n_delta_air', 'uc_n_delta_air']]

# remove things we don't need from pure water data
df_sceg_pure2 = df_sceg_pure[['molec_id', 'local_iso_id', 'quanta', 
                              'nu', 'uc_nu',
                              'sw', 'uc_sw',
                              'elower', 'uc_elower', 
                              'gamma_self', 'uc_gamma_self',
                              'n_self', 'uc_n_self',
                              'sd_self', 'uc_sd_self',
                              'delta_self', 'uc_delta_self', 
                              'n_delta_self', 'uc_n_delta_self']]



df_sceg = df_sceg_pure2.merge(df_sceg_air2, left_index=True, right_index=True)


f = open(os.path.join(d_sceg_save,'df_sceg_all.pckl'), 'wb')
pickle.dump([df_sceg], f)
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




#%% export to par file - NEED TO UPDATE TO INCLUDE THE NON-HITRAN PARAMETERS IN HEADER FILE

f = open(os.path.join(d_sceg_save,'df_sceg_all.pckl'), 'rb')
[df_sceg] = pickle.load(f)
f.close()

par_name = 'H2O'

# needs to match the header file that you are loading in
extra_params = {"n_self": "%7.4f", 
            	"sd_self": "%9.6f",    
            	"delta_self": "%9.6f",
                "n_delta_self": "%7.4f",
            	"sd_air": "%9.6f", 
            	"n_delta_air": "%7.4f"}


db.df_to_par(df_sceg, par_name, extra_params=extra_params, save_dir=d_sceg_save, print_name=False)


# need to fix error where the file adds " 1 12 ...." instead of just " 12 ..." at the beginning of each line
# did a find and replace and resolved the issue
# becomes apparent when hapi won't read in the file (lines parsed = 0)


#%% test the line list that you just saved


pld.db_begin('data - sceg')  


which_file = '1300 K 600 T'

d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water'
d_load = os.path.join(d_meas, which_file + ' bg subtraction.pckl')   

f = open(d_load, 'rb')
# what you get (first two): trans_BGremoved = (meas / bl) - H2O_bg, trans_BGincluded = (meas / bl) 
[trans_BGremoved, _, wavenumbers, T_meas, P_meas, y_h2o_meas, pathlength, _, fitresults, model2020, _, _] = pickle.load(f)
f.close() 

[istart, istop] = td.bandwidth_select_td(wavenumbers, [6650,7550], max_prime_factor=5)

wvn = wavenumbers[istart:istop]
trans_meas = trans_BGremoved[istart:istop]

TD_meas = np.fft.irfft(-np.log(trans_meas))
  
mod, pars = td.spectra_single_lmfit(sd=True)

pars['mol_id'].value = 1 # water = 1 (hitran molecular code)
pars['pathlength'].set(value = pathlength, vary = False) # pathlength in cm
pars['pressure'].set(value = P_meas / 760, vary = True, max=3) # pressure in atm (converted from Torr)
pars['molefraction'].set(value = y_h2o_meas, vary = True) # mole fraction

pars['temperature'].set(value = 300, vary = True) # temperature in K
TD_model_300 = mod.eval(xx=wvn, params=pars, name='H2O') # used to check baseline decision
trans_model_300 = np.exp(-np.real(np.fft.rfft(TD_model_300)))

pars['temperature'].set(value = T_meas, vary = True) # temperature in K
TD_model_1300 = mod.eval(xx=wvn, params=pars, name='H2O') # used to check baseline decision
trans_model_1300 = np.exp(-np.real(np.fft.rfft(TD_model_1300)))

weight = td.weight_func(len(wvn), 40, etalons = [])

trans_meas_baseline = np.exp(-np.real(np.fft.rfft(TD_meas - (1-weight) * (TD_meas - trans_model_1300))))
plt.plot(wvn, trans_meas)
plt.plot(wvn, trans_model_1300)
plt.plot(wvn, trans_model_300)
plt.plot(wvn, trans_meas_baseline)

plt.figure()
plt.plot(wvn, trans_model_1300)
plt.plot(wvn, trans_model_300)
plt.plot(wvn, trans_meas_baseline)
plt.plot(wvn, trans_meas_baseline - trans_model_1300 + 1.01)


            
fit_results = mod.fit(TD_meas, xx=wvn, params=pars, weights=weight, name='H2O')
td.plot_fit(wvn, fit_results, plot_td = True)
























