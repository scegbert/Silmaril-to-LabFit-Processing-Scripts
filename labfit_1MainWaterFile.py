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

r'''

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
bin_name = 'B49a'
d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files

iter_labfit = 1

prop_which = 'delta_air'

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
if bin_name_old == bin_name: lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], 2, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
# else: lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
# plt.title(bin_name)

bin_name_old = bin_name

asdfsadfsadf


df_calcs[df_calcs.index == 7810].delta_air


lab.save_file(d_labfit_main, bin_name, 'updated - pure water values', d_folder_input=d_labfit_kernal)


lab.plot_spectra(T,wvn,trans,res,res1, df_calcs[df_calcs.ratio_max>ratio_min_plot], 5, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)


       
#%% look for good yH2O features and features for exploring double power law relationships

d_old = r'E:\water database\air water' # for comparing to original input files


bin_names_test = ['B10', 
                  'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 
                  'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 
                  'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39', 'B40',
                  'B41', 'B42', 'B43', 'B44', 'B45', 'B46', 'B47']

# features_strong = {}

for i, bin_name in enumerate(bin_names_test): 
  
    bin_name = bin_name + 'a' 
   
    d_og = os.path.join(d_old, bin_name, bin_name + '-000-og') # for comparing to original input files
    
    [_, use_which] = lab.newest_rei(os.path.join(d_old, bin_name), bin_name)
    d_load = os.path.join(d_old, bin_name, use_which)[:-4]
    
    prop_which = False
    prop_which2 = False
    
    # [_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_old, bins, bin_name, og=True) # <-------------------
    
    [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name, d_load=d_load) # <-------------------
    df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_og, d_load=d_load) # <-------------------
    # features_strong[bin_name] = df_calcs[df_calcs.ratio_max>1.7].index.tolist()
    lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>0], 2, props[prop_which], props[prop_which2], features=features_strong[i], axis_labels=False) # <-------------------
    plt.title(bin_name)



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
    
r'''

#%% Calculated shifts and widths at each temperature for manual temperature dependence 


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


features_strong_flat = [item for sublist in features_strong for item in sublist]

for doublet in features_doublets: 
    if doublet != []: 
        for sub_doublet in doublet: 
            features_strong_flat.remove(sub_doublet[1])

T_conditions = list(dict.fromkeys([i.split()[0] for i in d_conditions]))

output_dpl = np.zeros((2,len(features_strong_flat), len(T_conditions), 8)) # values at each temperature
output_lab = np.zeros((2,len(features_strong_flat), 12)) # values as originally calculated by Labfit

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
                        
                                asdfasdfasdfasdfasdfsadf                            
                        
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
                        
                        
                        
    
    



#%% Set SD = 0


d_labfit_kernal = d_labfit_kp3
d_old_all = r'E:\water database' # for comparing to original input files

lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)


bin_names_all = bin_names.copy()

try_again = []

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

        d_og = os.path.join(d_old_all, d_type+' water', bin_name, bin_name + '-000-og') # for comparing to original input files
        # df_calcs = lab.information_df(False, bin_name, bins, cutoff_s296, T, d_old=d_og, d_load=d_load_file[:-4]) # <-------------------

        # save updated file
        open(os.path.join(d_labfit_kernal, bin_name) + '.inp', 'w').writelines(inp_saved)
        
        # run updated file
        feature_error = lab.run_labfit(d_labfit_kernal, bin_name, time_limit=60) # need to run one time to send INP info -> REI
        
        if feature_error is None: 
            
            # save SD = 0
            d_save_name = 'SD = 0 (all other floats included)'
            lab.save_file(d_load_folder, bin_name, d_save_name, d_folder_input=d_labfit_kernal, num_file=-1)

            # # plot SD = 0
            # [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
            # df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
            # lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>0], 2, props['n_'+d_which], props['sd_self'], axis_labels=False) # <-------------------
            # plt.title(bin_name)
            
        else: try_again.append(bin_name)
            
            
        







                        
                        
                        
#%%  revert delta_air back to HITRAN, update SD to 0.13, and update y_h2o



y_h2o_old = [0.0189750, 0.0191960, 0.0192650, 0.0193170, 0.0193940, 0.0194900, 0.0195320, 0.0194730, 
             0.0193570, 0.0193190, 0.0192070, 0.0192580, 0.0195700, 
             0.0189490, 0.0190160, 0.0189890, 0.0189220, 0.0189220, # duplicates
             0.0186050, 0.0189100, 0.0187070, 0.0185840, 0.0185690, 
             0.0191550, 0.0195360, 0.0192420, 0.0187510, 0.0188580, 
             0.0193090] # calculated using 38 features (listed above) using HITRAN 2020

y_h2o_new = [0.0194565, 0.0197830, 0.0199033, 0.0199502, 0.0200456, 0.0200617, 0.0200454, 0.0199243, 
             0.0198340, 0.0199169, 0.0198419, 0.0198157, 0.0200947, 
             0.0193546, 0.0195809, 0.0196135, 0.0195629, 0.0195001, 
             0.0189080, 0.0193659, 0.0193067, 0.0192536, 0.0191974, 
             0.0195506, 0.0199976, 0.0198608, 0.0194793, 0.0195815, 
             0.0200991] # calculated using 38 features (listed above) using updated database (~0.0001 lower)

y_h2o_lab = [0.0195687, 0.0198780, 0.0199699, 0.0199530, 0.0200125, 0.0200252, 0.0198926, 0.0197421, 
             0.0198567, 0.0198780, 0.0197901, 0.0197584, 0.0199763, 
             0.0196421, 0.0195510, 0.0196035, 0.0195262, 0.0194346, 
             0.0192477, 0.0194570, 0.0192889, 0.0192088, 0.0191539, 
             0.0193705, 0.0203911, 0.0199124, 0.0195472, 0.0196069, 
             0.0202362] # calculated using 251 features using LabFit
             
             
             
lines_main_header = 3 # number of lines at the very very top of inp and rei files
lines_per_asc = 134 # number of lines per asc measurement file in inp or rei file
lines_per_feature = 4 # number of lines per feature in inp or rei file (5 if using HTP - this version is untested)

lines_header_lwa = 18 # number of lines per header in lwa file

# d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'
d_labfit_main = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit'
d_labfit_k1 = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Labfit - Kiddie Pool'


d_labfit_kernal = d_labfit_main 

# d_old = r'H:\water database\air water' # for comparing to original input files
d_old = r'E:\water database\air water' # for comparing to original input files




for bin_name in bin_names: 
       
    d_og = os.path.join(d_old, bin_name, bin_name + '-000-og') # for comparing to original input files
    # d_og = os.path.join(d_old, bin_name, bin_name + '-000-HITRAN') # for comparing to original input files
    
    
    # copy file and load
    
    [_, use_which] = lab.newest_rei(os.path.join(d_old, bin_name), bin_name)
    
    inp_updated = open(os.path.join(d_old, bin_name, use_which), "r").readlines() 
    
    print('\n{}        {} features in this file, starting with {}'.format(bin_name, inp_updated[0].split()[3], use_which))
    
    inp_HT = open(os.path.join(d_old, bin_name, bin_name + '-000-HITRAN.rei'), "r").readlines()
    
    lines_until_features = lines_main_header + int(inp_updated[0].split()[2]) * lines_per_asc # all of the header down to the spectra
    
    
    # update yh2o
    
    for i_asc in range(int(inp_updated[0].split()[2])): 
    
        asc_name = ' '.join(inp_updated[lines_main_header+i_asc*lines_per_asc].split()[0].replace('_', ' ').split('.')[0].split()[1:-1])
            
        i_condition = d_conditions.index(asc_name)
        
        y_asc = float(inp_updated[lines_main_header+i_asc*lines_per_asc + 20].split()[0])
        
        y_h2o_old_i = y_h2o_old[i_condition]
        
        if y_h2o_old[i_condition] != y_h2o_old_i: throw_error = different_y
        
        y_h2o_new_i = y_h2o_lab[i_condition]
        
        inp_updated[lines_main_header+i_asc*lines_per_asc + 20] = '     {0:.6f}    '.format(y_h2o_new_i) + inp_updated[lines_main_header+i_asc*lines_per_asc + 20][17:]
        
    # update SD and delta (and n_delta back to 1)
    
    i_feature_updated = 0
    
    for i_feature_HT in range(int(inp_HT[0].split()[3])): 
       
        i_feature_HT += 1
        if str(i_feature_HT) != inp_HT[lines_until_features + lines_per_feature*(i_feature_HT-1)].split()[0]: throw_error = pleasehere
        
        delta_HT = inp_HT[lines_until_features + lines_per_feature*(i_feature_HT-1)][73:] # includes delta, n_delta, MW
        delta_HT = delta_HT[:12] + '1' + delta_HT[13:] # reset n_delta to 1 for all features
        
        i_feature_updated += 1
        line_updated = lines_until_features + lines_per_feature*(i_feature_updated-1)
        
        i_feature_updated_down = i_feature_updated

        # make sure you're on the right line
        while inp_updated[line_updated].split()[0] != str(i_feature_HT): 
            
            if inp_updated[line_updated+1][67:-1] == '0.12000': 
                inp_updated[line_updated+1] = inp_updated[line_updated+1][:67] + '0.13000\n' # update SD wherever you are
            
            i_feature_updated += 1 # look at the next feature
            line_updated = lines_until_features + lines_per_feature*(i_feature_updated-1)
            
            if inp_updated[line_updated].split()[0] != str(i_feature_HT): 
                
                i_feature_updated_down -= 1 # also look at above where you were
                line_updated = lines_until_features + lines_per_feature*(i_feature_updated_down-1)
                
                if inp_updated[line_updated].split()[0] == str(i_feature_HT): 
                    i_feature_updated = i_feature_updated_down # if we found it going down, use that index
        
        if inp_updated[line_updated].split()[0] != str(i_feature_HT): please = stop_here
            
        inp_updated[line_updated] = inp_updated[line_updated][:73] + delta_HT
        
        
        
        if inp_updated[line_updated+1][67:-1] == '0.12000': 
            inp_updated[line_updated+1] = inp_updated[line_updated+1][:67] + '0.13000\n' # update SD          
            
        if inp_updated[line_updated-lines_per_feature+1][67:-1] == '0.12000':     
            inp_updated[line_updated-lines_per_feature+1] = inp_updated[line_updated-lines_per_feature+1][:67] + '0.13000\n' # update SD above just in case
        
        if inp_updated[line_updated-2*lines_per_feature+1][67:-1] == '0.12000': 
            inp_updated[line_updated-2*lines_per_feature+1] = inp_updated[line_updated-2*lines_per_feature+1][:67] + '0.13000\n' # update SD 2 above just in case
        
        if inp_updated[line_updated-3*lines_per_feature+1][67:-1] == '0.12000': 
            inp_updated[line_updated-3*lines_per_feature+1] = inp_updated[line_updated-3*lines_per_feature+1][:67] + '0.13000\n' # update SD 3 above just in case
        # don't go down so you don't mess up the constraints
    

    # unfloat any floated shifts
    for i_feature_updated in range(int(inp_updated[0].split()[3])): 
 
        line_updated = lines_until_features + lines_per_feature*i_feature_updated    
 
        inp_updated[line_updated+2] = inp_updated[line_updated+2][:18] + '1  1' + inp_updated[line_updated+2][22:]
 
    
    
    use_which_updated = bin_name + '-100-updated yh2o SD and delta for last iteration.rei'
    
    open(os.path.join(d_old, bin_name, use_which_updated), 'w').writelines(inp_updated)
    
    







