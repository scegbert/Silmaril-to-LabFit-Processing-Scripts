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
offset = 2 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

# %% run specific parameters and function executions



cutoff_s296 = 1E-24 

bin_name = 'B12' # name of working bin (for these calculations)
d_labfit_kernal = d_labfit_main # d_labfit_main # d_labfit_kp # d_labfit_kp2









d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files

# use_rei = True

# features_doublets_remove = []
# features_remove_manually = []

prop_which2 = False
prop_which3 = False

nudge_sd = True
features_reject_old = []


print('\n\n\n     ******************************************')
print('     *************** using bin {} ******************       '.format(bin_name))
if d_labfit_kernal == d_labfit_kp: print('************** using KP Labfit folder **************')
if d_labfit_kernal == d_labfit_kp2: print('************** using KP #2 Labfit folder **************')
elif d_labfit_kernal == d_labfit_main: print('************** using MAIN Labfit folder **************')
print('     ******************************************\n\n\n')

please = stophere


# %% update n parameters to match Paul (Labfit default is 0.75 for all features)

# lab.nself_initilize(d_labfit_main, base_name_pure, n_update_name)


# %% mini-script to get the bin started, save OG file
lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions)

lab.run_labfit(d_labfit_kernal, bin_name) # <-------------------

lab.save_file(d_labfit_main, bin_name, d_og=True, d_folder_input=d_labfit_kernal) # make a folder for saving and save the original file for later

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

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
a_features_check = [int(x) for x in list(df_calcs[df_calcs.ratio_max>0].index)]
# print(a_features_check)

lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = a_features_check, axis_labels=False) # <-------------------
plt.title(bin_name)

# lab.plot_spectra(T,wvn,trans,res,False, False, offset, features = False) # don't plot the feature names

# lab.save_file(d_labfit_main, bin_name, d_save_name= 'reduced Chebyshev order for smaller regions', d_folder_input=d_labfit_kernal)


# %% fix giant features if the residual is throwing off neighbors

lab.float_lines(d_labfit_kernal, bin_name, features_start, props['sw'], 'rei_saved', [], d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
lab.float_lines(d_labfit_kernal, bin_name, features_start, props['nu'], 'inp_new', []) # INP -> INP, testing two at once (typically nu or n_self)
lab.float_lines(d_labfit_kernal, bin_name, features_start, props['gamma_self'], 'inp_new', []) # INP -> INP, testing two at once (typically nu or n_self)

lab.wait_for_kernal(d_labfit_kernal)

lab.run_labfit(d_labfit_kernal, bin_name) # <------------------

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------

feature_error = None
iter_labfit = 2

i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True) 
    
    df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------


[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = features_start, axis_labels=False) # <-------------------
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

# %% MAIN SEGMENT (will need to snag prop_which and feature parameters from txt file)


###############

bins_delta = {'B6': [4913],  
			 'B7': [5450, 5509],
			 'B8': [5655, 5665, 5763, 5850, 6004], 
			 'B9': [6283, 6305, 6360, 6373, 6382, 6618, 6643, 6647], 
			 'B10': [6872, 6917, 6919, 6981, 7013, 7021, 7094, 7114, 7183, 7284, 7328, 7335, 7414, 7442, 7466, 7469], 
			 'B11': [7649, 7677, 7749, 7807, 7823, 7830, 7905, 7912, 8051, 8085, 8123, 8138, 8220, 8244, 8268], 
			 'B12': [8291, 8297, 8404, 8450, 8581, 8582, 8609, 8623, 8632, 8637, 8639, 8658, 8686, 8724, 8752, 8764, 8793, 8841, 8889], 
			 'B13': [8942, 8948, 9037, 9057, 9066, 9068, 9097, 9177, 9259, 9286, 9298, 9402, 9452, 9460, 9486, 9491, 9532, 9566, 9579, 9594, 9626], 
			 'B14': [9715, 9738, 9739, 9751, 9803, 9809, 9866, 9887, 9908, 9918, 9972, 10032, 10046, 10072, 10088, 10151, 10165, 10169, 10227, 10230, 10268, 10274, 10340, 10350, 10372], 
			 'B15': [10456, 10470, 10477, 10495, 10559, 10624, 10684, 10716, 10779, 10790, 10827, 10875, 10890, 10934, 10948, 10954], 
			 'B16': [10983, 11028, 11052, 11056, 11063, 11124, 11152, 11208, 11221, 11247, 11353, 11487, 11708, 11716, 11721, 11733], 
			 'B17': [11819, 11929, 11956, 12006, 12048, 12070, 12252, 12464, 12510], 
			 'B18': [12589, 12590, 12611, 12647, 12707, 12732, 12837, 12856, 12864, 12865, 12871, 12881, 12884, 12893, 12895, 12916, 12922, 12923, 12933, 12940, 12943, 12952, 13000, 13021, 13023, 13034, 13048, 13073, 13114, 13121], 
			 'B19': [13256, 13271, 13286, 13288, 13298, 13304, 13383, 13413, 13509, 13525, 13543, 13545, 13550, 13589, 13610, 13634, 13646, 13673, 13764, 13775, 13822, 13824, 13848, 13867, 13873, 13886, 13921, 13950, 13965, 13981, 14005], 
			 'B20': [14045, 14060, 14089, 14093, 14110, 14127, 14139, 14153, 14169, 14172, 14179, 14227, 14275, 14286, 14304, 14436, 14459, 14473, 14475, 14509, 14547, 14566, 14591, 14624, 14664, 14740, 14742, 14817, 14845], 
             'B21': [14886, 14932, 14946, 14958, 14960, 15028, 15035, 15052, 15077, 15093, 15097, 15114, 15118, 15135, 15144, 15146, 15154, 15155, 15164, 15194, 15206, 15228, 15248, 15305, 15326, 15357, 15394, 15438, 15443, 15456, 15460, 15487, 15535, 15577, 15596, 15640], 
			 'B22': [15722, 15725, 15744, 15807, 15812, 15841, 15917, 15926, 15983, 15986, 15987, 16049, 16147, 16160, 16162, 16168, 16182, 16217, 16223, 16259, 16295, 16298, 16315, 16326, 16416, 16512, 16518, 16525, 16551, 16558, 16572], 
			 'B23': [16667, 16681, 16731, 16735, 16739, 16774, 16842, 16849, 16866, 16916, 17112, 17170, 17225, 17277, 17290, 17295, 17300, 17304, 17309, 17339, 17364, 17383, 17423, 17428, 17454, 17457, 17461, 17468], 
			 'B24': [17521, 17545, 17570, 17611, 17633, 17649, 17660, 17677, 17744, 17757, 17766, 17856, 17910, 17955, 17965, 17976, 18011, 18013, 18036, 18088, 18094, 18095, 18103, 18213], 
			 'B25': [18339, 18351, 18362, 18394, 18398, 18406, 18478, 18491, 18532, 18542, 18555, 18611, 18617, 18633, 18649, 18652, 18668, 18741, 18742, 18754, 18763, 18794, 18851, 18885, 18904, 18905, 18974, 19055, 19073, 19084, 19091, 19101, 19114, 19134], 
			 'B26': [19207, 19222, 19262, 19281, 19286, 19310, 19333, 19339, 19346, 19368, 19389, 19398, 19406, 19432, 19463, 19526, 19556, 19595, 19604, 19691, 19707, 19721, 19765, 19780, 19787, 19799, 19825, 19829, 19863, 19880, 19898], 
			 'B27': [19981, 20012, 20076, 20096, 20130, 20163, 20180, 20225, 20250, 20268, 20276, 20283, 20286, 20307, 20312, 20315, 20320, 20349, 20354, 20385, 20429, 20470, 20535, 20550, 20573, 20585, 20626, 20672, 20752, 20765, 20795, 20803, 20812, 20821, 20835, 20861, 20864], 
			 'B28': [20910, 20957, 21006, 21019, 21035, 21075, 21116, 21176, 21189, 21230, 21242, 21271, 21298, 21331, 21353, 21361, 21394, 21404, 21430, 21455, 21459, 21480, 21484, 21526, 21555, 21556, 21562, 21572, 21586],
			 'B29': [21652, 21691, 21739, 21873, 21991, 22269, 22372, 22431, 22491],
			 'B30': [22535, 22552, 22611, 22620, 22642, 22645, 22654, 22659, 22707, 22719, 22809, 22845, 22855, 22863, 22893, 22954, 23026, 23069, 23120, 23161, 23184, 23200, 23218, 23225, 23250, 23277, 23279, 23302, 23311, 23325, 23360, 23374, 23378, 23402, 23428, 23435, 23437], 
             'B31': [23499, 23507, 23610, 23615, 23714, 23854, 23877, 23916, 23942, 24128],
			 'B32': [24240, 24279, 24281, 24301, 24313, 24324, 24329, 24421, 24435, 24484, 24506, 24512, 24541, 24566, 24575, 24605, 24642, 24703, 24795, 24802, 24854, 24864, 24933, 24951],
			 'B33': [25065, 25093, 25143, 25151, 25176, 25210, 25233, 25239, 25246, 25333, 25394, 25446, 25459, 25483, 25495, 25502, 25509, 25534, 25549, 25552, 25573, 25625, 25695, 25710, 25720, 25732, 25820],
			 'B34': [25931, 25935, 26046, 26067, 26109, 26134, 26190, 26263, 26337, 26365, 26425, 26463, 26481, 26487, 26555, 26578, 26611],
			 'B35': [26664, 26691, 26745, 26750, 26804, 26810, 26905, 26908, 27034, 27107, 27181, 27194, 27207, 27227, 27244, 27268, 27282, 27291, 27307, 27334, 27343, 27348, 27376, 27442, 27468, 27475, 27487],
			 'B36': [27592, 27622, 27646, 27657, 27698, 27730, 27775, 27782, 27804, 27839, 27847, 28032, 28117, 28124, 28152, 28162, 28173, 28187, 28196, 28205, 28246, 28276, 28303], 
			 'B37': [28429, 28492, 28497, 28511, 28517, 28535, 28543, 28594, 28609, 28647, 28724, 28736, 28749, 28805, 28820, 28824, 28836, 28844, 28850, 28972, 29027, 29054, 29069, 29100, 29134, 29143, 29174, 29177, 29235, 29265, 29287, 29308, 29323, 29343], 
			 'B38': [29396, 29433, 29465, 29468, 29511, 29524, 29550, 29559, 29589, 29604, 29649, 29689, 29694, 29696, 29716, 29721, 29743, 29840, 29855, 29903, 29907, 29912, 29937, 29946, 30005, 30022, 30033, 30152, 30162, 30174, 30190, 30206, 30227, 30244, 30272, 30278], 
			 'B39': [30364, 30400, 30425, 30434, 30455, 30474, 30495, 30520, 30579, 30598, 30630, 30631, 30639, 30692, 30697, 30717, 30753, 30759, 30771, 30781, 30851, 30880, 30953, 30999, 31006, 31051, 31076, 31135, 31174, 31240], 
			 'B40': [31330, 31355, 31379, 31467, 31504, 31552, 31553, 31555, 31556, 31565, 31591, 31654, 31669, 31680, 31741, 31761, 31778, 31821, 31848, 31873, 31880, 31992, 31996, 32034, 32054, 32056],
             'B41': [32116, 32145, 32164, 32204, 32215, 32244, 32253, 32267, 32277, 32287, 32308, 32374, 32382, 32435, 32453, 32506, 32566, 32634, 32707, 32732, 32757, 32762, 32767, 32786, 32795, 32805, 32847, 32870, 32893], 
			 'B42': [32940, 32943, 32958, 33067, 33094, 33133, 33172, 33177, 33197, 33209, 33229, 33233, 33275, 33330, 33334, 33347, 33354, 33406, 33449, 33503, 33536, 33596, 33603, 33612], 
			 'B43': [33706, 33757, 33765, 33786, 33801, 33811, 33843, 33858, 33912, 33939, 33941, 33956, 34011, 34018, 34020, 34111, 34128, 34147, 34228, 34243, 34245, 34320, 34323, 34360], 
			 'B44': [34403, 34413, 34491, 34508, 34521, 34537, 34564, 34569, 34592, 34593, 34596, 34617, 34662, 34763, 34814, 34834, 34892, 34917, 34962, 34977, 34992, 34995, 35005, 35034, 35036], 
			 'B45': [35070, 35195, 35251, 35293, 35348, 35414, 35446, 35554, 35562, 35563, 35584, 35597, 35676, 35704], 
			 'B46': [35745, 35753, 35844, 35850, 35870, 35887, 35926, 35998, 36029, 36081, 36113, 36143, 36172, 36194, 36300, 36323, 36333], 
			 'B47': [36383, 36387, 36400, 36406, 36502, 36562, 36618, 36627, 36779, 36834, 36893], 
			# 'B48' - nothing
			 'B49': [37634, 37651, 37820, 37992], 
			 'B50': [38119],
             'B51': [38711], 
			# 'B52' - nothing
			# 'B53' - nothing
			# 'B54' - nothing
			# 'B55' - nothing
            }



bin_name = 'B20'


d_labfit_kp2 = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool 2'
d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

d_labfit_kernal = d_labfit_kp2 # d_labfit_main # d_labfit_kp # d_labfit_kp2


d_labfit_main = r'D:\OneDrive - UCB-O365\water database' # this is where all the files are kept right now
features_doublets = []
prop_whiches = [['delta_self', False, False, 'after delta self - updated',              False, False],
				['n_delta_self', False, False, 'after n delta self - updated', 'use_accepted', False]]

features_test = bins_delta[bin_name]

d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files

###############

# make sure all doublets are floated (avoids errors if we pause and then run in the night)
lab.float_lines(d_labfit_kernal, bin_name, features_test, props['nu'], 'rei_saved', features_doublets, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
# lab.run_labfit(d_labfit_kernal, bin_name) # make sure constraints aren't doubled up

lab.wait_for_kernal(d_labfit_kernal)

df_iter = {} # where we save information from the floating process
feature_error = None # first iteration with a feature that throws an error
feature_error2 = None # second iteration with the same feature throwing an error (something is up, time to stop)
already_reduced_i = False # if we reduce to avoid error, indicate here

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
            
            if already_reduced_i: please = stop_here
            already_reduced_i = True

        # if we timed out immidately, let's sniff stuff out
        elif (feature_error == 'no LWA' or feature_error == 'timeout') and i==1: 
                       
            sniff_features, sniff_good_reject, sniff_bad, sniff_iter = lab.feature_sniffer(features, d_labfit_kernal, bin_name, bins, prop_which, props, props_which, prop_which2,
                                                      iter_sniff=15, unc_multiplier=1.2, d_labfit_main=d_labfit_main)
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
            
        # default number of times to iterate 
        else: 
            
            if already_reduced_i and len(features_reject) == 0: # if you just ran a short version but didn't find anything
                please = stophere # you're going in circles
                
            iter_labfit = 10
            unc_multiplier = 1  

            already_reduced_i = False

        features_reject = []; feature_error = None # reset these guys for this round of testing
     
        
        #-- section to use if labfit crashed ------------------------------------------------------------------------
        
        
        lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which], 'rei_saved', features_constrain, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
        if prop_which2 is not False: lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which2], 'inp_new', features_constrain) # INP -> INP, testing two at once (typically nu or n_self)
        if prop_which3 is not False: lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which3], 'inp_new', features_constrain) # INP -> INP, testing two at once (typically sd_self)
        
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

            [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old, plots = False) # read results into python
            df_iter[d_save_name].append([df_compare, df_props, features]) # save output in a list (as a back up more than anything)
                    
            for prop_i in props_which: 
                if prop_i == 'sw': prop_i_df = 'sw_perc' # we want to look at the fractional change
                else: prop_i_df = prop_i
                
                features_reject.extend(df_props[df_props['uc_'+prop_i_df] > props[prop_i][4] * unc_multiplier].index.values.tolist())
                
                if prop_i == 'gamma_self' or prop_i == 'sd_self': 
                    
                    features_reject.extend(df_props[(df_props[prop_i_df] < 0.05) & (df_props['uc_'+prop_i_df] > df_props[prop_i_df])].index.values.tolist())
               
                # try: features_reject.extend((features_reject,df_props[df_props['uc_'+prop_i_df] > props[prop_i][4]].index.values.tolist())[1])
                # except: pass
                    
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


done = butnoplot


# get comparative information and plot change in each parameter
[df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old) # read results into python

# plot the new spectra with old residual included
[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)

# lab.save_file(d_labfit_main, bin_name, 'ditched unstable width floats', d_folder_input=d_labfit_kernal)

#%% check widths for weird things

[df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props['gamma_self'], props['n_self'], props['sd_self'], d_old=d_old) # read results into python



#%% re-run fits to fix something wrong

iter_labfit = 3

# prop_which = 'gamma_self'
# lab.float_lines(d_labfit_kernal, bin_name, features, props[prop_which], 'inp_new', features_constrain, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out

# prop_which = 'sw'
# lab.float_lines(d_labfit_kernal, bin_name,  [], props[prop_which], 'inp_new', [], d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out

print('     labfit iteration #1')
feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=90) # need to run one time to send INP info -> REI

i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True, time_limit=90) 

# [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props[prop_which], props[prop_which2], props[prop_which3], d_old=d_old) # read results into python

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)


if feature_error is None: lab.save_file(d_labfit_main, bin_name, 'final - updated', d_folder_input=d_labfit_kernal)
# if feature_error is None: lab.save_file(d_labfit_main, bin_name, 'ditched SD 36303 36306', d_folder_input=d_labfit_kernal)


#%% ditch floats that won't impact the shifts we are wanting to focus on



lab.unfloat_lines(d_labfit_kernal, bin_name, features_keep, features_keep_doublets, d_folder_input=d_labfit_main)

sdfasdf

lab.float_lines(d_labfit_kernal, bin_name, [feature for doublet in features_keep_doublets for feature in doublet],
                props['sw'], 'inp_new', features_keep_doublets) # INP -> INP, testing two at once (typically nu or n_self)

sdfasdfsad

print('     labfit iteration #1')
feature_error = lab.run_labfit(d_labfit_kernal, bin_name) # need to run one time to send INP info -> REI

iter_labfit = 5          
i = 1 # start at 1 because we already ran things once
while feature_error is None and i < iter_labfit: # run X times
    i += 1
    print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
    feature_error = lab.run_labfit(d_labfit_kernal, bin_name, use_rei=True) 

lab.save_file(d_labfit_main, bin_name, 'ditched floats', d_folder_input=d_labfit_kernal)

#%% feature sniffer (find which feature(s) is the problem)

lab.wait_for_kernal(d_labfit_kernal, minutes = 6) # delay longer than normal to save for last

# prop_which = 'n_self'

# features_test = [21652, 21691, 21699, 21739, 21772, 21778, 21873, 21920, 21929, 21933, 21937, 21979, 21991, 22025, 
#                  22060, 22158, 22167, 22246, 22256, 22269, 22319, 22327, 22337, 22366, 22372, 22389, 22422, 22431, 
#                  22455, 22466, 22491]

sniff_good, sniff_good_but_unc, sniff_bad, sniff_dict = lab.feature_sniffer(features_test, d_labfit_kernal, bin_name, bins, prop_which, props, props_which, 
                                                      iter_sniff=15, unc_multiplier=1, d_labfit_main=d_labfit_main)







# %% compare non-OG files

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=None) # <-------------------

lab.plot_spectra(T,wvn,trans,res,res6, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, axis_labels=False) # <-------------------
plt.title(bin_name)


# %% redo temp. dep. of the self shift with improved exponent guess


d_folder_output = r'D:\OneDrive - UCB-O365\water database'

n_delta_old = r'0.0000000  0.00000000    0.00000    0.00000    0.' # kernal to replace
n_delta_new = r'0.0000000  1.00000000    0.00000    0.00000    0.' # kernal that has been replaced


for bin_name in bin_names: 

    d_folder_bin = os.path.join(d_folder_output, bin_name)
    
    i = 0
    file_extension = ''
    file_name= ''
    
    try: 
        while file_extension != '.rei' or 'after delta self' not in file_name: # find file where we worked on delta_self
            i-=1
            file_name = os.listdir(d_folder_bin)[i]
            file_extension = file_name[-4:]
            
        file_name = os.listdir(d_folder_bin)[i-4] # go back 4 more to get the file right before the delta_self file
        if file_name[-4:] != '.rei': throw = anerrorplease # make sure it's an rei file
        
        # rei_all_old = open(os.path.join(d_folder_bin, file_name), "r").read() # load in the rei file right before delta_self
                
        # rei_all_new = rei_all_old.replace(n_delta_old, n_delta_new) # update n_delta
        
        # [num_file, _] = lab.newest_rei(d_folder_bin, bin_name) # get the newest file number for saving
        
        # d_save_name = 'updating n_delta_self initial value from 0 to 1'
        # d_output = os.path.join(d_folder_bin, bin_name + '-' + str(num_file+1).zfill(3) + '-' + d_save_name + '.rei') # avoid over writing existing files by adding 1 to file name
                
        # open(d_output, 'w').write(rei_all_new)
        
        # print('file saved as: ' + bin_name + '-' + str(num_file+1).zfill(3) + '-' + d_save_name)
        
    except: 
                
        print('after delta self file was not found for bin ' + bin_name) # if it didn't even run delta_self, there were no features to test







