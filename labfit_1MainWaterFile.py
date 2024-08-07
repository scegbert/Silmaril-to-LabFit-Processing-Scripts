r'''

labfit1 - main water file

main file for processing things in labfit (calls labfit, sets up the files to be processed by the labfit fortran engine)


r'''



import subprocess

import numpy as np
import matplotlib.pyplot as plt

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
# d_labfit_main = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit'

d_labfit_kp1 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP1'
d_labfit_kp2 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP2'
d_labfit_kp3 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP3'
d_labfit_kp4 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP4'
d_labfit_kp5 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP5'
d_labfit_kp6 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP6'
d_labfit_kp7 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - KP7'



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

d_labfit_kernal = d_labfit_main # d_labfit_kp1




# d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files
d_old_holder = r'H:\water database\pure water'
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


please = stophere

r'''
# %% update n parameters to match Paul (Labfit default is 0.75 for all features)

# lab.nself_initilize(d_labfit_main, base_name_pure, n_update_name)


# %% mini-script to get the bin started, save OG file

for bin_name in bin_names: 
    
    print(bin_name)

    lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions)

    lab.run_labfit(d_labfit_kernal, bin_name) # <-------------------
    
    lab.save_file(d_old_holder, bin_name, d_og=True, d_folder_input=d_labfit_kernal) # make a folder for saving and save the original file for later

# %% more intense script to get the bin started by testing which files allow labfit to run, save OG file

# lab.wait_for_kernal(d_labfit_kernal)

# which_ASC_files = lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions, ASC_sniffer = True)

which_ASC_files = ['2207_500_K_8_T_7.asc']

good_files = []
bad_files = []

print('\n\n\n\n\n')

for ASC_sniffer in which_ASC_files: 

    print(ASC_sniffer)    

    lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions, ASC_sniffer=ASC_sniffer)
    result = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # only let Labfit try running for one minute
    
    if result != 'timeout':
        print('     good\n')
        good_files.append(ASC_sniffer)
    else: 
        print('     bad\n')
        bad_files.append(ASC_sniffer)


# %% start list of things to float, plot stuff


lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions)

lab.run_labfit(d_labfit_kernal, bin_name) # <-------------------

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
a_features_check = [int(x) for x in list(df_calcs[df_calcs.ratio_max>0].index)]
# print(a_features_check)

features_new = df_calcs[(df_calcs.ratio_max>-1)&(df_calcs.index>1e6)].nu.to_frame() # list of index of new features
df_calcs.sort_index()
features_new['closest'] = 0
for index in features_new.index: 
    nu = features_new.loc[index].nu
    features_new.closest.loc[index] = (abs(df_calcs[df_calcs.index<1e6].nu - nu + 0.1)).idxmin()
    
    print('{}   {}    {}'.format(index, nu, features_new.closest.loc[index]))

lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = a_features_check, axis_labels=False) # <-------------------
plt.title(bin_name)

# lab.plot_spectra(T,wvn,trans,res,False, False, offset, features = False) # don't plot the feature names

#%% remove some features from the features_test 


features_keep = [x for x in features_test if x not in features_ditch]

# %% fix giant features if the residual is throwing off neighbors

prop_which = 'delta_air'
lab.float_lines(d_labfit_kernal, bin_name, features_delta_air, props[prop_which], 'rei_saved', [], 
                d_folder_input=d_labfit_main, nudge_delta_air=True, features_new=features_new) # float lines, most recent saved REI in -> INP out
# lab.float_lines(d_labfit_kernal, bin_name, features_delta_air, props['n_delta_air'], 'inp_new', [], d_folder_input=d_labfit_main, nudge_delta_air=True) # float lines, most recent saved REI in -> INP out


# lab.wait_for_kernal(d_labfit_kernal, minutes=1)

print('     labfit iteration #1') 
lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # <------------------

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------

feature_error = None
iter_labfit = 10

i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True, time_limit=60) 
    
    df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------


[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], features = features_delta_air, axis_labels=False) # <-------------------
plt.title(bin_name)

d_save_name = 'monster features that mess up their neighbors'

if feature_error is None: lab.save_file(d_labfit_main, bin_name, d_save_name, d_folder_input=d_labfit_kernal)


#%% figure out how much to shrink features that you can't see


print(lab.shrink_feature(df_calcs[df_calcs.index.isin(features_shrink)], cutoff_s296, T))

sdfsdfsdf

feature_error = lab.run_labfit(d_labfit_kernal, bin_name) #, use_rei=True) 

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = features_shrink, axis_labels=False) # <-------------------
plt.title(bin_name)


# asdfsdfs

if feature_error is None: lab.save_file(d_labfit_main, bin_name, d_save_name= 'shrunk non-visible features', d_folder_input=d_labfit_kernal)


# %% add new features (if required)

lab.wait_for_kernal(d_labfit_kernal)

iter_sniff = 10

props_new_feature = deepcopy(props)
props_new_feature['nu'][4] = 3*props['nu'][4] 
props_new_feature['sw'][4] = 5*props['sw'][4]  # an extra allowance for features with SW and E" floated

good_big, good_reject_big, bad_big, dict_big = lab.feature_sniffer(features_new_big, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                                   iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='big_all')

good_small, good_reject_small, bad_small, dict_small = lab.feature_sniffer(feature_new_small, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                                   iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='small_all')

props_new_feature['sw'][4] = 3*props['sw'][4] # remove allowance when not floating E"

good_b_nsg, good_reject_b_nsg, bad_b_nsg, dict_b_nsg = lab.feature_sniffer(good_reject_big + bad_big, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                                   iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='big_nsg')

good_b_ns, good_reject_b_ns, bad_b_ns, dict_b_ns = lab.feature_sniffer(good_reject_b_nsg + bad_b_nsg, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                                   iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='big_ns')


good_s_nsg, good_reject_s_nsg, bad_s_nsg, dict_s_nsg = lab.feature_sniffer(good_reject_small + bad_small, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                                   iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='small_nsg')

good_s_ns, good_reject_s_ns, bad_s_ns, dict_s_ns = lab.feature_sniffer(good_reject_s_nsg + bad_s_nsg, d_labfit_kernal, bin_name, bins, 'sw', props_new_feature, props_which, 
                                                               iter_sniff=iter_sniff, unc_multiplier=3, d_labfit_main=d_labfit_main, new_type='small_ns')

b11_good_big = good_big
b12_good_b_nsg = good_b_nsg
b13_good_b_ns = good_b_ns

b21_good_small = good_small
b22_good_s_nsg = good_s_nsg
b23_good_s_ns = good_s_ns

b3_reject = good_reject_s_ns+bad_s_ns + good_reject_b_ns+bad_b_ns



lab.wait_for_kernal(d_labfit_kernal)
iter_labfit = 20

lab.add_features(d_labfit_kernal, bin_name, b11_good_big, use_which='rei_saved', d_folder_input=d_labfit_main, new_type='big_all')
lab.add_features(d_labfit_kernal, bin_name, b12_good_b_nsg, use_which='inp_new', d_folder_input=d_labfit_main, new_type='big_nsg')
lab.add_features(d_labfit_kernal, bin_name, b13_good_b_ns, use_which='inp_new', d_folder_input=d_labfit_main, new_type='big_ns')
lab.add_features(d_labfit_kernal, bin_name, b21_good_small, use_which='inp_new', d_folder_input=d_labfit_main, new_type='small_all')
lab.add_features(d_labfit_kernal, bin_name, b22_good_s_nsg, use_which='inp_new', d_folder_input=d_labfit_main, new_type='small_nsg')
lab.add_features(d_labfit_kernal, bin_name, b23_good_s_ns, use_which='inp_new', d_folder_input=d_labfit_main, new_type='small_ns')

feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit = 40) # <------------------

features_dict = {}
i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True) 
    [df_props, _] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props['sw'], plots=False) # read results into python            
    features_dict[i] = df_props
    
[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features=False, axis_labels=False, all_names=True) # <-------------------
plt.title(bin_name)

plt.plot(b11_good_big, 100+offset/2*np.ones_like(b11_good_big), 'mx', markersize=25)
plt.plot(b11_good_big, 100+offset/2*np.ones_like(b11_good_big), 'm+', markersize=25)
plt.plot(b11_good_big, 100+offset/2*np.ones_like(b11_good_big), 'm1', markersize=25)
plt.plot(b11_good_big, 100+offset/2*np.ones_like(b11_good_big), 'm2', markersize=25)
plt.plot(b11_good_big, 100+offset/2*np.ones_like(b11_good_big), 'mo', fillstyle='none', markersize=25)

plt.plot(b12_good_b_nsg, 100+offset/2*np.ones_like(b12_good_b_nsg), 'mx', markersize=25)
plt.plot(b12_good_b_nsg, 100+offset/2*np.ones_like(b12_good_b_nsg), 'm+', markersize=25)
plt.plot(b12_good_b_nsg, 100+offset/2*np.ones_like(b12_good_b_nsg), 'mo', fillstyle='none', markersize=25)

plt.plot(b13_good_b_ns, 100+offset/2*np.ones_like(b13_good_b_ns), 'mx', markersize=25)
plt.plot(b13_good_b_ns, 100+offset/2*np.ones_like(b13_good_b_ns), 'mo', fillstyle='none', markersize=25)


plt.plot(b21_good_small, 100+offset/2*np.ones_like(b21_good_small), 'mx', markersize=25)
plt.plot(b21_good_small, 100+offset/2*np.ones_like(b21_good_small), 'm+', markersize=25)
plt.plot(b21_good_small, 100+offset/2*np.ones_like(b21_good_small), 'm1', markersize=25)
plt.plot(b21_good_small, 100+offset/2*np.ones_like(b21_good_small), 'm2', markersize=25)

plt.plot(b22_good_s_nsg, 100+offset/2*np.ones_like(b22_good_s_nsg), 'mx', markersize=25)
plt.plot(b22_good_s_nsg, 100+offset/2*np.ones_like(b22_good_s_nsg), 'm+', markersize=25)

plt.plot(b23_good_s_ns, 100+offset/2*np.ones_like(b23_good_s_ns), 'mx', markersize=25)


plt.plot(b3_reject, 100+offset/2*np.ones_like(b3_reject), 'mo', fillstyle='none', markersize=25)


if feature_error is None: lab.save_file(d_labfit_main, bin_name, d_save_name='added new features', d_folder_input=d_labfit_kernal)


# %% make changes to features that don't look great to begin with

lab.float_lines(d_labfit_kernal, bin_name, features_sw, props['sw'], 'rei_saved', features_doublets, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
lab.float_lines(d_labfit_kernal, bin_name, features_nu, props['nu'], 'inp_new', features_doublets) # INP -> INP, testing two at once (typically nu or n_self)
lab.float_lines(d_labfit_kernal, bin_name, features_gamma, props['gamma_self'], 'inp_new', []) # INP -> INP, testing two at once (typically nu or n_self)


adfasdfas # update special shifts and changes in SW

iter_labfit = 10

# lab.wait_for_kernal(d_labfit_kernal)

print('     labfit iteration #' + str(1)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
feature_error = lab.run_labfit(d_labfit_kernal, bin_name)#, time_limit=40) # <------------------

i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True) 
    
    df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------


[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = features_sw, axis_labels=False) # <-------------------
plt.title(bin_name)

d_save_name = 'features that needed extra TLC before other floats'

# d_save_name = 'monster features that mess up their neighbors'

if feature_error is None: lab.save_file(d_labfit_main, bin_name, d_save_name, d_folder_input=d_labfit_kernal)


adfasdfas

# cutoff_s296 = 1.5E-24
# df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------

a_features_check = [int(x) for x in list(df_calcs[df_calcs.ratio_max>0].index)]
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = a_features_check, axis_labels=False) # <-------------------
plt.title(bin_name)

r'''

# %% MAIN SEGMENT (will need to snag prop_which and feature parameters from txt file)


# make sure all doublets are floated (avoids errors if we pause and then run in the night)
lab.float_lines(d_labfit_kernal, bin_name, features_test, props['nu'], 'rei_saved', features_doublets, 
                d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
# lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up

lab.wait_for_kernal(d_labfit_kernal)

df_iter = {} # where we save information from the floating process
feature_error = None # first iteration with a feature that throws an error
feature_error2 = None # second iteration with the same feature throwing an error (something is up, time to stop)
already_reduced_i = False # if we reduce to avoid error, indicate here
already_sniffed = False

# features_test = features to try fitting
# features_doublets = doublets to try constraining

# features = features we are floating in this iteration
# features_constrain = doublets we are constraining in this iteration
# features_reject = features we are rejecting in this iteration
# features_doublets_reject = doublets we are rejecting in this iteration

features_remove = [] # features that threw an error (need to be removed and avoided)
features_remove_manually = [] # = features that don't meet uncertainty requirements. consider going back and unfloating (or keep if float was needed)

# run through a bunch of scenarios
for [prop_which, prop_which2, prop_which3, d_save_name, continuing_features, ratio_min_float] in prop_whiches: 
    
    iter_prop = 0 # how many times we have iterated on this property (ie attempts to repeatedly run labfit using a given set of features)
    iter_labfit_reduced = 20 # place holder to be updated and used later
    
    if continuing_features is not False: 

        # try again with rejected features (ie those that failed nu+sw could pass only sw
        if continuing_features == 'use_rejected': 
            features_test = a_features_reject.copy()
            features_doublets = a_features_doublets_reject.copy()            
        
        # try again with a different property (ie sw then nu)
        elif continuing_features == 'use_attempted': 
            pass # nothing needs to be done (features_test and features_doublets aren't modified during iteration)
        
        # if doing a progressive fit, only keep the good features (ie you need a delta_self to get an n_delta_self)
        elif continuing_features == 'use_accepted': 
            features_test = a_features.copy()
            features_doublets = a_features_constrain.copy()


    features_doublets_reject = []; feature_error2 = None # reset error checkers

    # reject all features below given threshold (why waste time letting them error out if we know it won't work?)
    if ratio_min_float is not False: 
        
        features_pre_test = features_test.copy()
        features_pre_doublets = features_doublets.copy()
        
        try: 
            [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
        except: 
            lab.float_lines(d_labfit_kernal, bin_name, [], props[prop_which], 'rei_saved', [], d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
            print('     labfit pre-run to get feature threshold')
            lab.run_labfit(d_labfit_kernal, bin_name)
            [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
            
        df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
        
        features_pre_reject = list({int(x) for x in list(df_calcs[df_calcs.ratio_max<ratio_min_float].index)} & set(features_pre_test))
        
        # remove them from the doublets list as well
        for doublet in features_doublets[:]: # if both doublets were rejected, ditch it
            if doublet[0] in features_pre_reject and doublet[1] in features_pre_reject: 
                features_pre_reject.extend(doublet)
                features_doublets.remove(doublet)
                features_doublets_reject.append(doublet)
            
            elif doublet[0] in features_pre_reject: # if only one is too small, might as well keep them both
                features_pre_reject.remove(doublet[0])
            
            elif doublet[1] in features_pre_reject: 
                features_pre_reject.remove(doublet[1])

        # new list of features to float (don't float any that is below threshold)
        features_test = list(set(features_pre_test) - set(features_pre_reject)) 
        
        print('\n\nthe following features were rejected before testing because they are below the ratio threshold\n')
        for feature in features_pre_reject: print(feature)
        print('\nthere are still {} features remaining to float\n\n'.format(len(features_test)))
        
    # otherwise pass: float the input features or the same ones as last iteration
    features = features_test.copy()
    features.sort()
    
    features_constrain = features_doublets.copy()
    features_reject = [0] # make sure we get in the while loop
    
    df_iter[d_save_name] = [[features_test.copy(), features_doublets.copy()]]; 
    
    already_sniffed = False
    
    while len(features_reject) > 0 or feature_error is not None or iter_prop <= 2 or already_reduced_i is True: # until an iteration doesn't reject any features (or feature_error repeats itself)
        
        iter_prop += 1
    
        # if this is the first iteration, let's get the easy ones out of the way 
        # fewer iterations and use unc_multiplier to give easier acceptance - ditch features that aren't even close
        if iter_prop == 1: 
            iter_labfit = 1
            unc_multiplier = 1.1
                        
        # if you're not floating anything, don't bother looping through things as intensely
        elif features == []: 
            iter_labfit = 0
            unc_multiplier = 1
        # if LWA or timeout error (didn't run) try to run one fewer times and remove offending feature
        elif (feature_error == 'no LWA' or feature_error == 'timeout') and i>1: 
            iter_labfit = i-1
            iter_labfit_reduced = iter_labfit
            unc_multiplier = 1        
            
            already_reduced_i = True

        # if we timed out immidately, let's sniff stuff out
        elif ((feature_error == 'no LWA' or feature_error == 'timeout') and (
                    i==1 or (already_reduced_i and not already_sniffed))) or (
                        already_reduced_i and len(features_reject) == 0 and already_sniffed is False): 
            
            already_sniffed = True     
            already_reduced_i = False
            
            sniff_features, sniff_good_reject, sniff_bad, sniff_iter = lab.feature_sniffer(features, d_labfit_kernal, bin_name, bins, prop_which, 
                                                                                           props, props_which, prop_which2,iter_sniff=10, 
                                                                                           unc_multiplier=1.2, d_labfit_main=d_labfit_main, 
                                                                                           features_new=features_new)
            features_sniffed = features.copy()
            features = sniff_features.copy()
            
            # check on the doublets
            for doublet in features_constrain[:]: 
                if doublet[0] in sniff_good_reject or doublet[1] in sniff_good_reject or doublet[0] in sniff_bad or doublet[1] in sniff_bad:
                    features_reject.extend(doublet)
                    features_constrain.remove(doublet)
                    features_doublets_reject.append(doublet)
                    
            # new list of features to float (don't float any that didn't work out)
            features = list(set(features) - set(features_reject)) 
            
            iter_labfit = 10
            unc_multiplier = 1   
            
         
        else: # we've already tried running things, sniffing things, all the things
            
            if already_reduced_i and len(features_reject) == 0: # if you just ran a short version but didn't find anything
                please = stophere # you're going in circles
                
            iter_labfit = 10
            unc_multiplier = 1  

            already_reduced_i = False

        features_reject = []; feature_error = None # reset these guys for this round of testing
     
        
        #-- section to use if labfit crashed ------------------------------------------------------------------------
        
        
        lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which], 'rei_saved', features_constrain, 
                d_folder_input=d_labfit_main, features_new=features_new) # float lines, most recent saved REI in -> INP out
        if prop_which2 is not False: lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which2], 'inp_new', features_constrain,
                                                     features_new=features_new) # INP -> INP, testing two at once (typically nu or n_self)
        if prop_which3 is not False: lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which3], 'inp_new', features_constrain, 
                                                     features_new=features_new) # INP -> INP, testing two at once (typically sd_self)
        
        print('     labfit iteration #1')
        feature_error = lab.run_labfit(d_labfit_kernal, bin_name) # need to run one time to send INP info -> REI        
        
        i = 1 # start at 1 because we already ran things once
        while feature_error is None and i < iter_labfit: # run X times
            i += 1
            print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True) 
            
            # [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old, plots = False) # read results into python


        #-- section to use if labfit crashed ------------------------------------------------------------------------
        
                                
        if feature_error is None: # if we made if through all iterations without a feature causing an error...
                    
            # df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # helpful for debugging but does slow things down
            try: 
                [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old, plots = False) # read results into python
    
                df_iter[d_save_name].append([df_compare, df_props, features]) # save output in a list (as a back up more than anything)
                        
                for prop_i in props_which: 
                    if prop_i == 'sw': prop_i_df = 'sw_perc' # we want to look at the fractional change
                    else: prop_i_df = prop_i
                    
                    features_reject.extend(df_props[df_props['uc_'+prop_i_df] > props[prop_i][4] * unc_multiplier].index.values.tolist())
                    
                    if prop_i == 'gamma_self' or prop_i == 'sd_self': 
                        
                        features_reject.extend(df_props[(df_props[prop_i_df] < 0.05) & (df_props['uc_'+prop_i_df] > df_props[prop_i_df])].index.values.tolist())
                   
            except: 
               
                rei_all = open(os.path.join(d_labfit_kernal, bin_name)+'.rei', "r").readlines()
                features_bad = [int(line.split()[0]) for line in rei_all if '*******' in line]
                
                for feature in features_bad: 
                    if feature in features: 
                        print('{} threw an error and was removed'.format(feature))
                    else: 
                        print('{} threw an error and could not be removed'.format(feature))
                        throw = errorplease
                
                features_reject.extend(features_bad)
                
                if len(features_bad) == 0: please = stophere # we didn't find anything and didn't set feature error to 0

        if feature_error is not None: 
            print('feature error - Labfit did not run due to struggles with feature ' + str(feature_error))
            
            if feature_error != 'timeout': 
                features_reject.append(feature_error) 
                features_remove.append(feature_error)
                
            else: 
                if iter_labfit_reduced == i-1: please = stophere # we're stuck in a staggered loop
                # run fully and get timeout, reduce iter_labfit and run without error, run fully again and timeout in same place...

            if feature_error2 == feature_error: please = stophere # things appear to be stuck in a loop            

        feature_error2 = feature_error # roll back the errors
        
        # check on the doublets (maybe we can float them as a pair), use a copy of list since we're deleting items from it
        for doublet in features_constrain[:]: 
            if doublet[0] in features_reject or doublet[1] in features_reject: # are any doublets in the reject list? (if should be fine, but fails sometimes. not sure why)
                features_reject.extend(doublet)
                features_constrain.remove(doublet)
                features_doublets_reject.append(doublet)
                
        # new list of features to float (don't float any that didn't work out)
        features = list(set(features) - set(features_reject)) 
        
        for featuresi in features_remove_manually: 
            # remove all instances of problem feature
            features_reject = [x for x in features_reject if x != featuresi] 
            
            try: features_reject.remove(featuresi)
            except: pass
        
        if features_reject_old == features_reject and features_reject != []: 
            # we're stuck in a loop, either stop the loop (throw error) or ignore those features
            features_remove_manually.extend(features_reject) 
            
        features_reject_old = features_reject.copy()
        print(features_reject)
        
    # generate output things for copying into notes
    a_features_reject = sorted(list(set(features_test).difference(set(features)))) # watch features that were removed  
    a_features = sorted(features) # list in a list for copying to notepad
    a_features_constrain = sorted(features_constrain)
    a_features_doublets_reject = sorted(features_doublets_reject)
    a_features_remove = sorted(features_remove)
    
    # save useful information from the fitting process into dictionary (won't overwrite with multiple iterations)
    df_iter[d_save_name].append([a_features.copy(), a_features_constrain.copy(), a_features_reject.copy(), a_features_doublets_reject.copy(), a_features_remove.copy()])
       
    # save file, this is what you will be reloading for the next round of the for loop  
    lab.save_file(d_labfit_main, bin_name, d_save_name, d_folder_input=d_labfit_kernal)

print(' *** these features were manually removed - you might need to remove them in the save rei file *** ')
print(features_remove_manually)
a_features_remove_manually = features_remove_manually

# add back in prop2 for plotting and analysis        
if d_save_name == 'sw cleanup after only nu after only sw': prop_which2 = 'nu'
elif d_save_name == 'after only sd_self': prop_which2 = 'n_self'; prop_which3 = 'gamma_self'
elif d_save_name == 'after n delta self': prop_which2 = 'delta_self'
elif d_save_name == 'after n delta self - updated': prop_which2 = 'delta_self'

if d_save_name == 'after only sd_air': prop_which2 = 'n_air'; prop_which3 = 'gamma_air'
elif d_save_name == 'after n delta air': prop_which2 = 'delta_air'


# done = butnoplot


# get comparative information and plot change in each parameter
[df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old) # read results into python

# plot the new spectra with old residual included
[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], 2, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)

finished = True


#%% re-run fits to fix something wrong


d_labfit_kernal = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit'
bin_name = 'B17'
d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files

iter_labfit = 1

prop_which = 'sw'

# prop_which = 'sd_self'
# features_test = [12589,12590] #, 10994, 10993]
# features_doublets = [[12589,12590]] #, [10994, 10993]]
# lab.float_lines(d_labfit_kernal, bin_name, features_test, props[prop_which], 'inp_new', features_doublets, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
# # 'inp_new'

if bin_name_old == bin_name: res1 = res.copy()
else: 
    lab.float_lines(d_labfit_kernal, bin_name, [], props[prop_which], 'inp_saved', [], d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out

print('     labfit iteration #1')
feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI

i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True, time_limit=60) 

# [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old) # read results into python

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
# if bin_name_old == bin_name: lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], 2, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------

lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)

bin_name_old = bin_name

asdfsadfsadf


df_calcs[df_calcs.index == 7810].delta_air


lab.save_file(d_labfit_main, bin_name, 'updated - pure water values', d_folder_input=d_labfit_kernal)


lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], 5, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)


       
#%% plotting for Lockheed

cutoff_s296 = 1.5e-24
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------


lab.plot_spectra(T,wvn,trans,res,False, df_calcs, offset, axis_labels=False) # <-------------------

plt.ylim(98.4, 102.96)
plt.xlim(6882.13, 6882.89)

plt.ylabel('Transmission   &   102+Residual')
plt.xlabel('Wavenumber [cm$^{-1}$]')

plt.savefig(r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\plots\1 labfithelp.png',bbox_inches='tight',dpi=600)


#%% look for good yH2O features and features for exploring double power law relationships

# d_old = r'E:\water database\air water' # for comparing to original input files
d_old = r'E:\water database\pure water' # for comparing to original input files

nu = 6969.069086

bin_names_test = ['B10', 
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
                  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
                  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47']


for key in bins.keys(): 
    if key != 'all': 
        if (bins[key][1] < nu) and (bins[key][2] > nu): 
            bin_names_test = [key]
            
print(bin_names_test)
print('\n\n\n\n\n\n\n')


ssssssssssssssssssssss


# features_strong = {}

for i, bin_name in enumerate(bin_names_test): 
  
    bin_name = bin_name # + 'a' 
   
    d_og = os.path.join(d_old, bin_name, bin_name + '-000-og') # for comparing to original input files
    
    [_, use_which] = lab.newest_rei(os.path.join(d_old, bin_name), bin_name)
    d_load = os.path.join(d_old, bin_name, use_which)[:-4]
    
    prop_which = False
    prop_which2 = False
    
    # [_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_old, bins, bin_name, og=True) # <-------------------
    
    [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name, d_load=d_load) # <-------------------
    df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_og, d_load=d_load) # <-------------------
    # features_strong[bin_name] = df_calcs[df_calcs.ratio_max>1.7].index.tolist()
    lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>-2], 2, axis_labels=False) # <-------------------
    # lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>0], 2, features=features_strong[i], axis_labels=False) # <-------------------
    plt.title(bin_name)
    

a = df_calcs[(df_calcs.uc_nu > -0.5)&(df_calcs.nu > nu-0.5)&(df_calcs.nu < nu+0.5)]
    
pausehere



lab.save_file(d_old, bin_name, 're-ran with updated yh2o', d_folder_input=d_labfit_kernal)



#%% calculate new SW that will be used for yh2o


bin_names_test = ['B10', 
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
                  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
                  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47']

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


features_strong_flat = [item for sublist in features_strong for item in sublist]

for doublet in features_doublets: 
    if doublet != []: 
        for sub_doublet in doublet: 
            features_strong_flat.remove(sub_doublet[1])


output_sw = np.zeros((len(d_conditions), len(features_strong_flat)))
output_uc_sw = np.zeros((len(d_conditions), len(features_strong_flat)))
output_sw_old = np.zeros((len(d_conditions), len(features_strong_flat)))
output_attempts = np.zeros((len(d_conditions), len(features_strong_flat)))

d_old = r'E:\water database\air water' # for comparing to original input files
lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)


for i_bin, bin_name in enumerate(bin_names_test):
    
    bin_name = bin_name + 'a' 
    
    d_og = os.path.join(d_old, bin_name, bin_name + '-000-og') # for comparing to original input files
    
    # get most up-to-date INP file into Labfit
    [_, use_which] = lab.newest_rei(os.path.join(d_old, bin_name), bin_name)
    d_load = os.path.join(d_old, bin_name, use_which)[:-4] 
    
    inp_updating = open(d_load + '.inp', "r").readlines()
    
    num_ASC = int(inp_updating[0].split()[2])
    num_features = int(inp_updating[0].split()[3])
    
    line_first_feature = lines_main_header + num_ASC * lines_per_asc # all of the header down to the spectra
    line_first_constraint = line_first_feature + num_features * lines_per_feature
    
    
    # remove all constraints and unfloat all features
    inp_updating[1] = inp_updating[1][:26] + '      0      0' + inp_updating[1][40:] # remove all constraints from file header
    inp_updating = inp_updating[:line_first_constraint] # remove all constraints from bottom of file
    
    for i in range(num_features): 
        
        inp_updating[line_first_feature+i*lines_per_feature+2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n' # unfloat everything
    
    inp_header = inp_updating[:lines_main_header]
    inp_features = inp_updating[line_first_feature:]
    
    
    for i_meas, meas_condition in enumerate(d_conditions): 
        
        # remove all measurement files except the one we're investigating
        num_ascs = 0
        for i_asc in range(num_ASC): 
            
            line_asc = lines_main_header+i_asc*lines_per_asc
            
            
            asc_name = ' '.join(inp_updating[line_asc].split()[0].replace('_', ' ').split('.')[0].split()[1:-1])
            
            if asc_name == meas_condition: 
                
                if num_ascs == 0: 
                    num_ascs+=1
                    inp_asc = inp_updating[line_asc:line_asc+lines_per_asc] # for first (and/or only) instance
                    
                else: 
                    num_ascs+=1
                    inp_asc.extend(inp_updating[line_asc:line_asc+lines_per_asc]) # if ultiple instances of this ASC file
        
        inp_header[0] = inp_header[0][:25] + '    {}'.format(num_ascs) + inp_header[0][30:]
        
        inp_updated = inp_header.copy()
        inp_updated.extend(inp_asc)
        inp_updated.extend(inp_features)
        
        open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
        
        print('\n************************            {}            {}\n'.format(bin_name, meas_condition))
        
        num_attempts = 1
        
        # float lines we're investigating (nu, sw, gamma, SD), constrain all values for doublets
        features = features_strong[i_bin]
        features_constrain = features_doublets[i_bin]
        
        lab.float_lines(d_labfit_kernal, bin_name, features, props['nu'], 'inp_new', features_constrain) 
        lab.float_lines(d_labfit_kernal, bin_name, features, props['sw'], 'inp_new', features_constrain) 
        lab.float_lines(d_labfit_kernal, bin_name, features, props['gamma_self'], 'inp_new', features_constrain) 
        lab.float_lines(d_labfit_kernal, bin_name, features, props['sd_self'], 'inp_new', features_constrain) 
           
        # run labfit
        print('     labfit iteration #1')
        feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=45) # need to run one time to send INP info -> REI
        
        if feature_error != None: 
            
            num_attempts+=1
            
            open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)
            
            lab.float_lines(d_labfit_kernal, bin_name, features, props['nu'], 'inp_new', features_constrain) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['sw'], 'inp_new', features_constrain) 
            lab.float_lines(d_labfit_kernal, bin_name, features, props['gamma_self'], 'inp_new', features_constrain) 
            
            # run labfit
            print('     labfit iteration #2 (no SD)')
            feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=45) # need to run one time to send INP info -> REI
            
            if feature_error != None: 
            
                num_attempts+=1    
            
                open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_updated)    
            
                lab.float_lines(d_labfit_kernal, bin_name, features, props['nu'], 'inp_new', features_constrain) 
                lab.float_lines(d_labfit_kernal, bin_name, features, props['sw'], 'inp_new', features_constrain) 
                
                # run labfit
                print('     labfit iteration #3 (no widths of any kind)')
                feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=45) # need to run one time to send INP info -> REI
        
        if feature_error == None: 
            
            # plot results (at least at first to make sure things aren't crazy)
            # [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
            [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props['sw'], d_old=d_og, plots=False) # read results into python
            # lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>0], 2, props['nu'], props['sw'], features=features, axis_labels=False) # <-------------------
            # plt.title(bin_name + ' - ' + meas_condition)
    
    
            # extract updated SW values and compile into dict/list
            for feat in features: 
                
                if feat in features_strong_flat: 
                
                    i_feat = features_strong_flat.index(feat)
                       
                    output_sw[i_meas, i_feat] = df_props.loc[feat].sw
                    output_uc_sw[i_meas, i_feat] = df_props.loc[feat].uc_sw
                    output_sw_old[i_meas, i_feat] = df_props.loc[feat].sw_old
                
                    output_attempts[i_meas, i_feat] = num_attempts
    



#%% Set SD = 0


d_labfit_kernal = d_labfit_kp4
d_old_all = r'E:\water database' # for comparing to original input files

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)


bin_names_all = bin_names.copy()

error_running = []
error_plotting = []

# put both air and pure items in the bins dict
if bin_names_all[0][-1] == 'a' and 'B1' not in bins.keys(): 
    bin_names2 = [item[:-1] for item in bin_names_all]
    
    for i in range(len(bin_names2)):   
        bins[bin_names2[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 
        if i == 0: bins[bin_names2[i]][0] = 0
        elif i == len(bin_names2)-1: bins[bin_names2[i]][-1] = 0

elif bin_names_all[0][-1] != 'a' and 'B1a' not in bins.keys(): 
    bin_names2 = [item+'a' for item in bin_names_all]
    
    for i in range(len(bin_names2)):   
        bins[bin_names2[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 
        if i == 0: bins[bin_names2[i]][0] = 0
        elif i == len(bin_names2)-1: bins[bin_names2[i]][-1] = 0


for i_bin, bin_name in enumerate(bin_names):

    # start with pure water data, no matter what the d_type was at the top of the file
    if bin_name[-1] == 'a': bin_name = bin_name[:-1]    
            
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
        
        # set all SD floats to 1 and all SD values to 0
        for i in range(num_features): 
            
            inp_saved[line_first_feature+i*lines_per_feature+1] = inp_saved[line_first_feature+i*lines_per_feature+1][:-11] + '   0.00000\n' # SD = 0
            inp_saved[line_first_feature+i*lines_per_feature+2] = inp_saved[line_first_feature+i*lines_per_feature+2][:-4] + '  1\n' # unfloat SD
                    
        # extract updated values and compile into dict/list
        [T, P, wvn, trans_og, res_og, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_load_folder, bins, bin_name, og=True) # <-------------------

        # save updated file
        open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_saved)
        
        # run updated file
        feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
        
        if feature_error is None: 

            try: 
                
                [_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_load_folder, bins, bin_name, og=True) # <-------------------
    
                [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
                df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
                lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>0], 2, props['n_'+d_which], props['sd_self'], axis_labels=False) # <-------------------
                plt.title(bin_name)
                
                # save SD = 0
                d_save_name = 'SD = 0 (all other floats included)'
                lab.save_file(d_load_folder, bin_name, d_save_name, d_folder_input=d_labfit_kernal, num_file=-1)
                
            except: error_plotting.append(bin_name)
            
        else: error_running.append(bin_name)
            



#%% final check for SD = 0 


bin_name = 'B46a' # haven't done 25a yet


if bin_name[-1] == 'a': 
    d_type = 'air'
    d_which = 'air'    
    cutoff_s296 = 5E-24 
    
else: 
    d_type = 'pure'
    d_which = 'self'
    cutoff_s296 = 1E-24 


offset = 2

d_load_folder = os.path.join(d_old_all, d_type+' water')
[_, use_which] = lab.newest_rei(os.path.join(d_load_folder, bin_name), bin_name)
d_load_SD = os.path.join(d_load_folder, bin_name, use_which)[:-4]

d_save_name = 'SD = 0 (all other floats included)'
d_load_noSD = os.path.join(d_load_folder, bin_name, bin_name + '-000-' + d_save_name)

[_, _,   _,     _, res_sd,      _,     _,           _] = lab.labfit_to_spectra(_, bins, bin_name, d_load=d_load_SD) # <-------------------

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(_, bins, bin_name, d_load=d_load_noSD) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_load_SD) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_sd, df_calcs[df_calcs.ratio_max>-1.5], offset, props['n_'+d_which], props['sd_self'], axis_labels=False) # <-------------------
plt.title(bin_name)


# change = df_calcs[abs(df_calcs.nu - df_calcs.nu_og) > 0.01].index.to_list()
# plt.title(bin_name + '    ' + ', '.join(map(str, change)))
# print(df_calcs[abs(df_calcs.nu - df_calcs.nu_og) > 0.01].nu)


#%% final touch ups for SD = 0


feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI

# feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60, use_rei=True) 

# save SD = 0
if feature_error == None: lab.save_file(d_load_folder, bin_name, d_save_name, d_folder_input=d_labfit_kernal, num_file=-1)


res_old = res.copy()

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(_, bins, bin_name, d_load=d_load_noSD) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_load_SD) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_old, df_calcs[df_calcs.ratio_max>-1.5], offset, props['n_'+d_which], props['sd_self'], axis_labels=False) # <-------------------
plt.title(bin_name)


# change = df_calcs[abs(df_calcs.nu - df_calcs.nu_og) > 0.01].index.to_list()
# plt.title(bin_name + '    ' + ', '.join(map(str, change)))
# print(df_calcs[abs(df_calcs.nu - df_calcs.nu_og) > 0.01].nu)



#%% Set some delta = 0

bin_names_test = ['B19a']

d_labfit_kernal = d_labfit_kp7
d_old_all = r'E:\water database' # for comparing to original input files

d_type = 'air'
d_which = 'air'    
cutoff_s296 = 5E-24 

offset = 2

d_load_folder = os.path.join(d_old_all, d_type+' water')

for bin_name in bin_names_test: 
    
    [_, use_which] = lab.newest_rei(os.path.join(d_load_folder, bin_name), bin_name)
    d_load_newest = os.path.join(d_load_folder, bin_name, use_which)[:-4]
        
    lab.float_lines(d_labfit_kernal, bin_name, [], props['nu'], 'inp_saved', [], d_folder_input=d_load_folder)




#%% exploring transitions for Tibor

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)

prop_which = 'nu'
prop_which2 = 'sw'

where_save = 'updated' # 'HITRAN' # 'updated'

# # intensity check transitions
# wvn_new = [7010.972238, 6924.605637, 6633.399489, 7155.824131, 7164.784714, 7243.570000, 6931.149824, 6788.644188, 7032.545215,
#             7150.799191, 6944.179118, 7033.030246, 7026.623336, 6821.589772, 7171.737815, 7026.716363, 6970.794222, 7327.595578, 
#             6881.232414, 7019.902243, 6707.432199, 7382.344828, 6739.801060, 7444.231575, 7378.922989, 6932.724849, 7274.873102, 
#             7390.262912, 7024.496499, 7163.954821, 7298.916203, 6963.081136, 6896.536731, 7212.596124, 7047.900105, 6698.637463, 
#             6856.267559, 6756.033820, 7137.650628, 7132.736817, 7008.534449, 7034.113240, 7071.360385, 7080.715018, 6990.753891, 
#             6829.879005, 7019.842821, 6866.666058, 7452.779382, 7046.844303, 6808.611156, 6967.859272, 7004.028540, 7384.306035, 
#             7033.320189, 6820.036726, 7177.801154, 6818.218863, 6812.879061, 6887.118988, 7071.936373, 7159.000000, 7390.260000, 
#             7000.343856, 6953.705813, 6848.281019, 6993.175574, 6831.475224, 6857.852666, 6857.850000, 7150.115996, 6907.976805, 
#             7160.450000, 6948.045385, 6806.939525, 6756.619933, 6755.920322, 6656.870372, 6983.005397, 7253.072635, 6813.989361, 
#             6846.313548, 6849.290013, 6666.690896, 7151.146591, 7048.477550, 7291.050935, 6781.570898, 6663.755814, 6881.638378, 
#             7302.222114, 6908.144317, 7291.141066, 7022.922097, 6712.458951, 6881.033994, 6783.366244, 7051.091577, 6990.583511, 
#             6831.889723, 7092.269839, 7115.949328, 7178.255437, 6845.088827, 7376.544992, 6736.962030, 6828.575357, 6898.123198, 
#             7049.611010, 6977.922732, 6989.155056, 7491.478353, 6996.549031, 6816.964224]

# elower_tibor = [5076.26601276, 3629.09611416, 5690.87960189, 5690.87960189, 5742.03692618, 4412.31708543, 5614.08735324, 5683.33256543,
#                 5674.30938274, 5587.51867639, 5586.59022201, 5421.26757229, 5534.11045464, 5484.00020847, 5621.33438267, 5500.85662039, 
#                 5475.75537710, 5355.26279349, 5256.38144166, 5554.83422372, 5579.49121917, 5477.00583794, 5475.75537710, 5632.04223517,
#                 5523.11739752, 5473.80321659, 5414.12677528, 3598.72694932, 5289.95905112, 5439.05630366, 5534.11045464, 4847.62209626,
#                 5538.80210440, 5324.66663070, 5442.09781133, 5512.01424456, 3940.54492808, 5464.33208073, 5400.73672878, 5496.98050248, 
#                 5409.54718184, 5418.80338201, 5430.18360443, 4329.49578805, 5258.86857730, 5457.36965218, 5334.98828289, 5271.37011549,
#                 5310.23977038, 4695.83624707, 5294.03719450, 5414.12677528, 5457.36965218, 4902.61402321, 5039.62741921, 5421.26757229, 
#                 5439.05630366, 5324.66663070, 5477.00583794 ,4735.84492592, 5300.17823670, 5258.63047053 ,3598.72694932 ,5027.25669634, 
#                 4052.81087277, 5204.00865187, 5369.69201430, 5336.32705442, 5074.22162349, 5074.22162349, 5334.98828289, 5378.74451935, 
#                 4381.73537579, 5612.49335386, 5289.95905112, 5342.18721646, 5096.24530746, 5399.33130374, 5276.46433772, 5213.26934644, 
#                 5300.17823670, 5255.34673811, 5369.69201430, 5207.80209252, 5316.80416753, 5273.63261357, 5289.15192694, 4285.64751251, 
#                 5316.80416753, 5355.26279349, 5421.26757229, 5279.67108511, 5289.15261747, 5193.45762501, 5152.96439205, 5256.44866157, 
#                 5193.88218190, 5193.88218190, 5256.84526905, 5255.34673811, 5207.80209252, 5256.44866157, 2983.39637988, 5175.95511953, 
#                 5193.88218190, 5204.98799958, 5108.34932253, 4150.28728186, 5122.39289211, 5070.03101818, 5208.89531551, 5688.50215558, 
#                 4992.12160894, 4842.13129829]


# # perfect match transitions
# wvn_new = [6684.995034, 6697.705177, 6740.191594, 6741.582445, 6751.306137, 6752.650963, 6760.503055, 6761.573774, 6794.575903, 
#            6797.251562, 6801.947130, 6811.080000, 6813.698899, 6818.723941, 6839.587789, 6842.703691, 6843.131625, 6844.526139, 
#            6860.874158, 6865.274078, 6875.119121, 6885.467660, 6899.540419, 6908.917939, 6993.456014, 7010.731433, 7020.162126, 
#            7022.966648, 7023.157017, 7029.840208, 7034.582213, 7061.041155, 7106.401573, 7128.616090, 7191.344022, 7228.794815, 
#            7238.074731, 7247.050000, 7251.218761, 7259.535631, 7272.861417, 7273.750704, 7278.165277, 7279.482835, 7287.755869, 
#            7291.141000, 7300.653860, 7305.433141, 7334.044800]

# elower_tibor = [4820.64274425, 5065.45355085, 5089.33620276, 5152.96439205, 5090.03869904, 4095.80321525, 5076.26601276, 4643.38034904, 
#                 5122.34750387, 4837.69957025, 5355.26279349, 5523.11739752, 4572.44628998, 5035.12651241, 5105.72971486, 5027.07397664, 
#                 4918.23474301, 5029.81049429, 5067.07675234, 5027.55961623, 4977.04419783, 3770.87886655, 4996.33159102, 3935.34465559, 
#                 4432.86017651, 4830.61439013, 5067.07675234, 5171.05987663, 4939.79424877, 5196.50054423, 5067.07675234, 4559.70763657, 
#                 4996.33159102, 4437.39977573, 3868.98696608, 4931.27155607, 5021.39143671, 5002.56867608, 4894.58572495, 5122.39289211, 
#                 4977.04419783, 3894.79890713, 5090.03869904, 4977.04419783, 5067.07675234, 5289.15261747, 5500.85662039, 5067.07675234, 
#                 5119.37559142]


# wavenumber check needed
wvn_new = [6656.261041, 6656.506836, 6680.030081, 6695.406495, 6711.304691, 6741.142758, 6745.720000, 6754.096308, 6756.335626, 6767.738772, 
            6781.020657, 6782.724159, 6783.360000, 6783.419051, 6786.170913, 6791.439003, 6795.200000, 6799.520825, 6805.238473, 6809.851824, 
            6811.080343, 6815.723782, 6818.937177, 6828.289101, 6830.936234, 6832.127416, 6835.347320, 6863.816093, 6865.752648, 6866.212608, 
            6870.370000, 6870.760000, 6870.764004, 6872.307952, 6872.920866, 6874.733555, 6881.525589, 6899.121124, 6915.248379, 6917.390000, 
            6918.627858, 6943.949098, 6953.916456, 6956.770000, 6965.417285, 6966.014182, 6973.138848, 6973.428826, 6975.470640, 6977.429180, 
            6983.862821, 6988.163287, 6988.558605, 6988.676265, 6988.749491, 6988.750000, 7001.308994, 7008.775435, 7016.067984, 7033.997484, 
            7034.989500, 7036.917390, 7052.680458, 7056.432036, 7063.280448, 7063.706514, 7067.366079, 7069.387226, 7081.962357, 7084.161227, 
            7085.560000, 7087.887129, 7093.967265, 7100.914199, 7101.750000, 7124.717440, 7147.249274, 7160.737101, 7162.994065, 7164.656387, 
            7165.013016, 7178.680000, 7183.320314, 7184.179063, 7184.888873, 7192.572784, 7195.955098, 7201.564335, 7202.057335, 7213.893172, 
            7214.319689, 7225.016172, 7231.756569, 7248.094169, 7250.308019, 7250.310000, 7268.809041, 7273.135067, 7280.780000, 7280.780505, 
            7282.280000, 7287.750000, 7300.771068, 7302.639187, 7305.529745, 7317.489642, 7334.044851, 7340.543818, 7382.340000, 7385.252401, 
            7389.400358, 7401.298403, 7415.060361, 7415.464579, 7438.092500, 7443.801993, 7451.888710, 7478.654807, 7490.228731, 7490.809659, 
            7491.417655, 7491.821286, 7492.599507, 7495.707905, 7495.959427]

elower_tibor = [5163.08124650, 5163.08763707, 4066.12257395, 4971.26066265, 5466.40248811, 4408.02882210, 4260.46692240, 4427.22682357,
                5208.89531551, 3526.62746419, 4784.99992542 ,4992.12160894, 5193.88218190 ,2927.07518877 ,5256.44866157, 4756.39401793, 
                5193.45762501, 5409.69797106, 4436.94052569, 5096.24538482, 5523.11739752, 5039.62741921, 5144.40901328, 4992.12160894, 
                5670.61527220, 2927.07518877, 5430.18360443, 5052.66862206, 5184.73428194, 4031.85357180, 4769.03845451, 4195.81808078, 
                4195.81808078, 5315.50009057, 4977.04419783, 4329.49578805, 4249.52442169, 5454.59620510, 5429.11838152, 3655.48281415, 
                4644.21717450, 5199.59660508, 4756.39401793, 4387.06285453, 4926.34694016, 5457.23699800, 4728.22116642, 4611.79479731, 
                4980.22313032, 4837.69957025, 5286.56288401, 5069.08825131, 4837.69957025, 4750.38772568, 4717.10447381, 4717.10447381, 
                5256.11644305, 4976.04327386, 4048.24954447, 5523.11739752, 5022.28126107, 5171.05987663, 5196.50054423, 5193.45762501, 
                5477.00583794, 4643.99911285, 5334.98828289, 4782.92004741, 5144.40901328, 5316.80416753, 5020.02614885 ,5429.11838152, 
                5169.03924855, 3752.41620098, 5286.56288401, 4506.73381954, 5399.33130374, 5169.03924855, 5238.38529808, 2724.16711089, 
                3567.25475111, 3951.31499959, 5524.56916455, 4427.22682357, 4525.23906839, 3032.69039203, 4918.23474301, 5119.37559142, 
                5455.88864111, 3895.25290074, 4612.79070416, 4759.85308121, 2952.39373221, 4268.24093181, 5466.40248811, 5466.40248811, 
                5015.70405468, 4976.04327386, 5334.86018233, 5334.86018233, 5310.23977038, 5067.07675234, 5409.69797106, 3144.57876256, 
                4696.83464688, 3535.87044665, 5119.37559142, 5208.89531551, 5477.00583794, 3482.48017553, 3136.41248609, 3386.05233763, 
                4752.73294407, 5324.66663070, 5255.44874185, 5611.32977444, 4062.83809343, 3647.45949024, 4738.62165700, 5246.07900130, 
                5163.08124650, 5604.30897052, 5713.24954725, 5031.97797906, 5471.86399333]


# # problematic cases
# wvn_new = [7390.262912, # 27
#            6856.267559, # 36
#            7080.715018, # 43
#            7046.844303, # 49
#            6887.118988, # 59
#            6953.705813, # 64
#            6953.705813, # 64
#            6953.705813, # 64
#            6781.570898, # 87
#            7178.255437, # 102
#            6898.123198, # 107
#            6963.081136] # 31         
           
# elower_tibor = [6149.493696, # 27
#                 6240.487063, # 36
#                 6013.447885, # 43
#                 6340.33789, # 49
#                 4735.844887, # 59
#                 5475.755373, # 64
#                 6705.593530, # 64
#                 5069.088248, # 64
#                 5827.338197, # 87
#                 5507.474906, # 102
#                 6290.183818, # 107
#                 6019.181143] # 31  


index_labfit = [0.0] * len(elower_tibor)

sw_updated = [0.0] * len(elower_tibor)
sw_unc_updated = [0.0] * len(elower_tibor)

nu_updated = [0.0] * len(elower_tibor)
nu_unc_updated = [0.0] * len(elower_tibor)

# for i_wvn, wvn in enumerate(wvn_new): 
    
i_wvn = 12
wvn = wvn_new[i_wvn]

   
print('           {}   (i={})  '.format(wvn,i_wvn))

# which bin is this transition in? 
bin_indices = []

for i_bin in range(len(bin_breaks) - 1):
    if bin_breaks[i_bin] <= wvn < bin_breaks[i_bin + 1]:
        bin_name = bin_names[i_bin]
        break

# read in og HITRAN info
d_old = os.path.join(d_old_holder, bin_name, bin_name + '-000-og') # for comparing to original input files
[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------

# load in updated file and run labfit to prep for next steps
lab.float_lines(d_labfit_kernal, bin_name, [], prop_which, use_which='inp_saved', d_folder_input=d_labfit_main)

lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up
[T, P, wvn_plot, trans, res_pre_change, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------   


# load in INP file and get ready to update E"
try: i_transition = int(df_calcs[(np.round(df_calcs.nu,3) == np.round(wvn,3))&(df_calcs.index>100000)].index[0])
except: 
    try: 
        i_transition = int(df_calcs[(np.round(df_calcs.nu,2) == np.round(wvn,2))&(df_calcs.index>100000)].index[0])
        print('\n wavenumber issues - check that this is the right transition \n')
    except: 
        i_transition = False
        print('\n wavenumber issues - could not find the right transition \n')

if i_transition is not False: 
    
    i_closest = int(df_calcs[df_calcs.nu == df_calcs.iloc[(df_calcs[(df_calcs.index<1e6)].nu-wvn).abs().argmin()].nu].index[0])
    
    if where_save == 'updated': [_, use_which] = lab.newest_rei(os.path.join(d_labfit_main, bin_name), bin_name)
    elif where_save == 'HITRAN': use_which = bin_name + '-000-HITRAN.inp'
    inp_latest = open(os.path.join(d_labfit_main, bin_name, use_which[:-4])+'.inp', "r").readlines()   
    
    lines_until_features = lines_main_header + int(inp_latest[0].split()[2]) * lines_per_asc # all of the header down to the spectra    
    line_searching = lines_until_features + lines_per_feature*i_closest    
    
    # update E" for the new transition
    if int(inp_latest[line_searching].split()[0]) == i_transition: # check if the transition is right where predicted (very, very unlikely)
        line_transition = line_searching  
    
    else:

        num_checked = 1
        line_found = False
        
        while line_found is False: 
       
            if int(inp_latest[line_searching+4*num_checked].split()[0]) == i_transition: 
                line_transition = line_searching + 4*num_checked
                line_found = True
              
            elif int(inp_latest[line_searching-4*num_checked].split()[0]) == i_transition:
                line_transition = line_searching - 4*num_checked
                line_found = True
               
            elif num_checked > 1000: stop_we_did_not_find_the_transition
            
            else: num_checked += 1
    
    inp_latest[line_transition] = inp_latest[line_transition][:52] + "{:.7f}".format(elower_tibor[i_wvn]) + inp_latest[line_transition][64:]
    
    elower_previous = df_calcs.loc[i_transition].elower
    s296_previous = df_calcs.loc[i_transition].sw
    s296_guess = s296_previous * lab.strength_T(1300, elower_previous, wvn) / lab.strength_T(1300, elower_tibor[i_wvn], wvn)
    
    inp_latest[line_transition] = inp_latest[line_transition][:29] + "{:.5E}".format(s296_guess) + inp_latest[line_transition][40:] # update S296
    
    
    # unfloat all other lines so we're only looking for this one transition right now    
    lines_until_features = lines_main_header + int(inp_latest[0].split()[2]) * lines_per_asc # all of the header down to the spectra    
    number_of_transitions = int(inp_latest[0].split()[3])
    
    for i in range(number_of_transitions): 
        line_latest = lines_until_features + lines_per_feature*i            
        if line_latest != line_transition: 
            inp_latest[line_latest+ 2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'
                
            
    open(os.path.join(d_labfit_kernal, bin_name)+'.inp', 'w').writelines(inp_latest)    
    
        
    # run labfit - only floating the new transition we're focused on 
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up

    offset = 0.5
    
    if feature_error is None: 
        
                
        [T, P, wvn_plot, trans, res_post_change, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
        df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------   
    
        lab.plot_spectra(T,wvn_plot,trans,res_post_change,res_pre_change, df_calcs[df_calcs.ratio_max>ratio_min_plot], 
                         offset, props[prop_which], props[prop_which2], axis_labels=False, res_extra=res_og) # <-------------------
        plt.title('wvn={} ({})   bin={} ({})'.format(wvn, i_wvn,bin_name, i_transition))
        plt.xlim(wvn-0.2, wvn+0.3)
        plt.ylim(99, 100+offset*3.7)
        
        
        
        plt.savefig(os.path.join(d_labfit_kernal,'plots', where_save, '24EgSyCoDr {} at {}.jpg'.format(i_wvn, wvn)))
        # plt.close()

        
        index_labfit[i_wvn] = i_transition
        
        sw_updated[i_wvn] = df_calcs.loc[i_transition].sw
        sw_unc_updated[i_wvn] = df_calcs.loc[i_transition].uc_sw
        
        nu_updated[i_wvn] = df_calcs.loc[i_transition].nu
        nu_unc_updated[i_wvn] = df_calcs.loc[i_transition].uc_nu
        
        
        wwwwwwwwwwwwwwwwwwwwwwwww

    
else: 
    
    lab.plot_spectra(T,wvn_plot,trans,res_pre_change,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], 
                          offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
    plt.title('FAILED wvn={:.2f} ({})   bin={} ({}) FAILED'.format(wvn, i_wvn,bin_name, i_transition))
    plt.xlim(wvn-0.25, wvn+0.25)
    plt.ylim(99, 101.7)
    
#     plt.savefig(os.path.join(d_labfit_kernal,'plots', where_save, '24EgSyCoDr {} at {} - FAILED.jpg'.format(i_wvn, wvn)))
#     plt.close()
    
    
#     index_labfit[i_wvn] = -1
        
#     sw_updated[i_wvn] = -1
#     sw_unc_updated[i_wvn] = -1
    
#     nu_updated[i_wvn] = -1
#     nu_unc_updated[i_wvn] = -1
        



#%% explore E"

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)

prop_which = 'nu'
prop_which2 = 'sw'

where_save = 'updated' # 'HITRAN' # 'updated'

e_lower_steps = np.linspace(-3000, 3000, 13) # np.linspace(-2000, 2000, 9)


# problematic cases
wvn_new = [7390.262912, # 27
           6856.267559, # 36
           7080.715018, # 43
           7046.844303, # 49
           6887.118988, # 59
           6953.705813, # 64
           6953.705813, # 64
           6953.705813, # 64
           6781.570898, # 87
           7178.255437, # 102
           6898.123198, # 107
           6963.081136] # 31         
           
elower_tibor = [6149.493696, # 27
                6240.487063, # 36
                6013.447885, # 43
                6340.33789, # 49
                4735.844887, # 59
                5475.755373, # 64
                6705.593530, # 64
                5069.088248, # 64
                5827.338197, # 87
                5507.474906, # 102
                6290.183818, # 107
                6019.181143] # 31  


index_labfit = [0.0] * len(elower_tibor)

sw_updated = [0.0] * len(elower_tibor)
sw_unc_updated = [0.0] * len(elower_tibor)

nu_updated = [0.0] * len(elower_tibor)
nu_unc_updated = [0.0] * len(elower_tibor)

# for i_wvn, wvn in enumerate(wvn_new): 
        
i_wvn = 4
wvn = wvn_new[i_wvn]

   
print('           {}   (i={})  '.format(wvn,i_wvn))

# which bin is this transition in? 
bin_indices = []

for i_bin in range(len(bin_breaks) - 1):
    if bin_breaks[i_bin] <= wvn < bin_breaks[i_bin + 1]:
        bin_name = bin_names[i_bin]
        break

# read in og HITRAN info
d_old = os.path.join(d_old_holder, bin_name, bin_name + '-000-og') # for comparing to original input files
[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------

# load in updated file and run labfit to prep for next steps
lab.float_lines(d_labfit_kernal, bin_name, [], prop_which, use_which='inp_saved', d_folder_input=d_labfit_main)

lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up
[T, P, wvn_plot, trans, res_pre_change, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------   


# load in INP file and get ready to update E"
try: i_transition = int(df_calcs[(np.round(df_calcs.nu,3) == np.round(wvn,3))&(df_calcs.index>100000)].index[0])
except: i_transition = False

if i_transition is not False: 
    
    i_closest = int(df_calcs[df_calcs.nu == df_calcs.iloc[(df_calcs[(df_calcs.index<1e6)].nu-wvn).abs().argmin()].nu].index[0])
    
    if where_save == 'updated': [_, use_which] = lab.newest_rei(os.path.join(d_labfit_main, bin_name), bin_name)
    elif where_save == 'HITRAN': use_which = bin_name + '-000-HITRAN.inp'
    inp_latest = open(os.path.join(d_labfit_main, bin_name, use_which[:-4])+'.inp', "r").readlines()   
    
    lines_until_features = lines_main_header + int(inp_latest[0].split()[2]) * lines_per_asc # all of the header down to the spectra    
    line_searching = lines_until_features + lines_per_feature*i_closest    
    
    # update E" for the new transition
    if int(inp_latest[line_searching].split()[0]) == i_transition: # check if the transition is right where predicted (very, very unlikely)
        line_transition = line_searching  
    
    else:

        num_checked = 1
        line_found = False
        
        while line_found is False: 
       
            if int(inp_latest[line_searching+4*num_checked].split()[0]) == i_transition: 
                line_transition = line_searching + 4*num_checked
                line_found = True
              
            elif int(inp_latest[line_searching-4*num_checked].split()[0]) == i_transition:
                line_transition = line_searching - 4*num_checked
                line_found = True
               
            elif num_checked > 1000: stop_we_did_not_find_the_transition
            
            else: num_checked += 1
           
    
    # unfloat all other lines so we're only looking for this one transition right now    
    lines_until_features = lines_main_header + int(inp_latest[0].split()[2]) * lines_per_asc # all of the header down to the spectra    
    number_of_transitions = int(inp_latest[0].split()[3])
    
    for i in range(number_of_transitions): 
        line_latest = lines_until_features + lines_per_feature*i            
        if line_latest != line_transition: 
            inp_latest[line_latest+ 2] = '   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'
 
    res_elower = [0] * len(e_lower_steps)
    
    for i_elower, elower_step in enumerate(elower_tibor[i_wvn]+e_lower_steps): 

        print('testing E" of {}'.format(elower_step))            

        inp_latest[line_transition] = inp_latest[line_transition][:52] + "{:.7f}".format(elower_step) + inp_latest[line_transition][64:] # update E"    
        
        elower_previous = df_calcs.loc[i_transition].elower
        s296_previous = df_calcs.loc[i_transition].sw
        s296_guess = s296_previous * lab.strength_T(1300, elower_previous, wvn) / lab.strength_T(1300, elower_step, wvn)
        
        inp_latest[line_transition] = inp_latest[line_transition][:29] + "{:.5E}".format(s296_guess) + inp_latest[line_transition][40:] # update S296
                
        open(os.path.join(d_labfit_kernal, bin_name)+'.inp', 'w').writelines(inp_latest)    
                   
        # run labfit - only floating the new transition we're focused on 
        feature_error = lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up
        
        if feature_error is None: 
        
            [T, P, wvn_plot, trans, res_elower[i_elower], wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
        
        else: 
        
            res_elower[i_elower] = False

    offset = 0.5
    # [T, P, wvn_plot, trans, res_elower, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------   
    lab.plot_spectra(T,wvn_plot,trans,res_og,res_pre_change, df_calcs[df_calcs.ratio_max>ratio_min_plot], 
                     offset, props[prop_which], props[prop_which2], axis_labels=False, res_extra=res_elower) # <-------------------
    
    plt.title('wvn={}, E" guess={}'.format(wvn, elower_tibor[i_wvn]))
    plt.xlim(wvn-0.2, wvn+0.3)
    plt.ylim(99, 100 + offset*(2.5+len(e_lower_steps)))
    
    fig = plt.gcf()
    fig.set_size_inches(5, 6)
    
    plt.annotate('HITRAN (no transition)',(wvn-0.18, 100+1*offset),color='k') 
    plt.annotate('Labfit {}cm-1'.format(int(elower_previous)),(wvn-0.18, 100+2*offset),color='k') 
    
    for i_elower, elower_step in enumerate(elower_tibor[i_wvn]+e_lower_steps):
        plt.annotate('E" {}cm-1'.format(int(elower_step)),(wvn-0.18, 100+(3+i_elower)*offset),color='k') 
    
    plt.vlines(wvn, 99.5, 100+(3+i_elower)*offset)
       
    plt.savefig(os.path.join(d_labfit_kernal,'plots', where_save, '24EgSyCoDr {} at {}.jpg'.format(i_wvn, wvn)))
            
    plt.close()






