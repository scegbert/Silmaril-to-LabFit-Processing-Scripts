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

d_labfit_kp2 = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool 2'
d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'


if d_type == 'pure': 
    base_name = 'p2020' + 'n_gam' # n update name
    d_cutoff_locations = d_labfit_main + '\\cutoff locations pure.pckl'

elif d_type == 'air': 
    base_name = 'p2020a_updated'
    d_cutoff_locations = d_labfit_main + '\\cutoff locations air.pckl'

ratio_min_plot = -2 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
offset = 2 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

# %% run specific parameters and function executions



cutoff_s296 = 5E-24 

bin_name = 'B34a' # name of working bin (for these calculations)
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

# %% MAIN SEGMENT (will need to snag prop_which and feature parameters from txt file)


# make sure all doublets are floated (avoids errors if we pause and then run in the night)
lab.float_lines(d_labfit_kernal, bin_name, features_test, props['nu'], 'rei_saved', features_doublets, 
                d_folder_input=d_labfit_main, features_new=features_new) # float lines, most recent saved REI in -> INP out
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
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)




#%% check widths for weird things

[df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, bins, bin_name, props_which, props['gamma_self'], props['n_self'], props['sd_self'], d_old=d_old) # read results into python


#%% setup for rerunning last check out


bin_name = 'B22'

d_labfit_kp2 = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool 2'
d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

# d_labfit_kernal = d_labfit_kp2 # d_labfit_main # d_labfit_kp # d_labfit_kp2

d_labfit_main = r'D:\OneDrive - UCB-O365\water database\done' # this is where all the files are kept right now
d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files


lab.float_lines(d_labfit_kernal, bin_name, [], props['nu'], 'rei_saved', [], d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out
feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=90) # need to run one time to send INP info -> REI

prop_which = 'delta_self'
prop_which2 = 'n_delta_self'

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], 5, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name + d_labfit_kernal[-13:])


#%% re-run fits to fix something wrong


iter_labfit = 2

res1 = res.copy()

# prop_which = 'gamma_self'
# lab.float_lines(d_labfit_kernal, bin_name, features_test, props[prop_which], 'inp_new', features_doublets, d_folder_input=d_labfit_main) # float lines, most recent saved REI in -> INP out

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

# [_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
# lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)


if feature_error is None: lab.save_file(d_labfit_main, bin_name, 'final - updated - again', d_folder_input=d_labfit_kernal)
# if feature_error is None: lab.save_file(d_labfit_main, bin_name, 'ditched SD 36303 36306', d_folder_input=d_labfit_kernal)


asdfsadfsadf

lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], 5, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)




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

#%% make new OG files that actually match HITRAN values


good_files= []
bad_files = []

base_name_HITRAN = 'p2020'

for bin_name in bins:
    
    if bin_name not in ['all']: 
    
        print(bin_name)
            
        lab.bin_ASC_cutoff(d_labfit_main, base_name_HITRAN, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions)
        result = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # only let Labfit try running for one minute
        
        if result != 'timeout':
            print('     good\n')
            good_files.append(bin_name)
        else: 
            print('     bad\n')
            bad_files.append(bin_name)
        
        

#%% save the new HITRAN og files in their folders

d_labfit_folders = r'D:\OneDrive - UCB-O365\water database'
failed = []

for bin_name in bins:

    if bin_name not in ['all']:     
    
        try: 
            lab.save_file(d_labfit_folders, bin_name, d_og=True, d_folder_input=d_labfit_kernal) # make a folder for saving and save the original file for later
            
        except: 
            try: 
                
                lab.save_file(os.path.join(d_labfit_folders,'done'), bin_name, d_og=True, d_folder_input=d_labfit_kernal) # make a folder for saving and save the original file for later
    
            except: 
                
                failed.append(bin_name)
    


