
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
props['elower'] = ['elower', 'E\"', 4, 34, 1e6] # using 1e6 for things we aren't floating
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
              7540.6, 7560.5, 7580.5, 7600.0, 7620.0, 7640.0, 7660.0, 7720.0, 7799.9]
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
# bins['all'] = [-buffer, 6700, 7158.3, buffer] 

d_labfit_kp2 = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool 2'
d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

base_name_pure = 'p2020'
d_cutoff_locations = d_labfit_main + '\\cutoff locations pure.pckl'

base_name_air = 'B2020Ja1'

if d_type == 'pure': base_name = base_name_pure + 'n'
elif d_type == 'air': base_name = base_name_air + 'n'

cutoff_s296 = 1E-24
ratio_min_plot = 0 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
ratio_min_fit = 1.5
offset = 2 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

nudge_sd = True
features_reject_old = []


prop_whiches = [[          'nu',     'sw',     False,            'nu + sw',          False, False],
				[  'gamma_self', 'n_self', 'sd_self',   'after all widths', 'use_accepted', False], 
				[  'delta_self',    False,     False,   'after delta self', 'use_accepted', False]]

# %% run specific parameters and function executions

d_labfit_kernal = d_labfit_kp2 # d_labfit_main # d_labfit_kp # d_labfit_kp2

# bin_name = 'B25' # name of working bin (for these calculations)

sdfsdfsdf



# for bin_name in bin_names[33:37]: #[::-1]:
   
bin_name = 'B36'
   
d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files


# %% mini-script to get the bin started, save OG file

lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_kernal, bins, bin_name, d_cutoff_locations, d_conditions)
lab.run_labfit(d_labfit_kernal, bin_name) # <-------------------

lab.save_file(d_labfit_main, bin_name, d_og=True, d_folder_input=d_labfit_kernal) # make a folder for saving and save the original file for later


# %% start list of things to float, plot stuff

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------

df_calcs_overlap = df_calcs[df_calcs.ratio_max > ratio_min_fit-1]

df_calcs_overlap2 = pd.DataFrame()
df_calcs_overlap2['deltaL'] = df_calcs_overlap.nu.diff(periods=1).abs().fillna(value=1)
df_calcs_overlap2['deltaR'] = df_calcs_overlap.nu.diff(periods=-1).abs().fillna(value=1)

df_calcs_overlap['delta_min'] = df_calcs_overlap2.values.min(axis=1)

df_calcs_clean = df_calcs_overlap[(df_calcs_overlap.delta_min > 0.04) & (df_calcs_overlap.ratio_max > ratio_min_fit)]



features_test = [int(x) for x in list(df_calcs_clean.index)]
features_doublets = []

# lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = features_test, axis_labels=False, all_names=False) # <-------------------
# plt.title(bin_name)


# %% mini-script to check fits for specified features (will need to snag prop_which and feature parameters from notes txt file)




df_iter = {} # where we save information from the floating process
feature_error = None # first iteration with a feature that throws an error
feature_error2 = None # second iteration with the same feature throwing an error (something is up, time to stop)

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
    
    iter_prop = 1
    
    if continuing_features is not False: 

        # try again with rejected features (ie those that failed nu+sw could pass only sw
        if continuing_features == 'use_rejected': 
            features_test = a_features_reject.copy()
            features_doublets = a_features_doublets_reject.copy()
            
            # features_test.append(12105)
            # features_test.append(12109)
            
        
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
            lab.run_labfit(d_labfit_kernal, bin_name)
            [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
            
        df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
        
        features_pre_reject = list({int(x) for x in list(df_calcs[df_calcs.ratio_max<ratio_min_float].index)} & set(features_pre_test))
        
        # remove them from the doublets list as well
        for doublet in features_doublets[:]: 
            if doublet[0] in features_pre_reject or doublet[1] in features_pre_reject: # are any doublets in the reject list? (if should be fine, but fails sometimes. not sure why)
                features_pre_reject.extend(doublet)
                features_doublets.remove(doublet)
                features_doublets_reject.append(doublet)

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
    
    while len(features_reject) > 0 or feature_error is not None or iter_prop <= 2: # until an iteration doesn't reject any features (or feature_error repeats itself)
                
        # if this is the first iteration, let's get the easy ones out of the way 
        # fewer iterations and use unc_multiplier to give easier acceptance - ditch features that aren't even close
        if iter_prop == 1: iter_labfit = 2; unc_multiplier = 1.5
        # if you're not floating anything, don't bother looping through things as intensely
        elif features == []: iter_labfit = 0; unc_multiplier = 1
        # default number of times to iterate 
        else: iter_labfit = 10; unc_multiplier = 1

        features_reject = []; feature_error = None # reset these guys for this round of testing
     
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
                        
            # if i == 9: 
            #     [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------

            #     df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # helpful for debugging but does slow things down
            #     asdfsdfsdf
                
        if feature_error is None: # if we made if through all iterations without a feature causing an error...
            
            iter_prop += 1 # after we get 2 and ditch the "not close" ones, then we'll start the real, longer iterations
        
            # df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # helpful for debugging but does slow things down

            [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, d_old, bins, bin_name, props[prop_which], props_which, props[prop_which2], plots = False) # read results into python
            df_iter[d_save_name].append([df_compare, df_props, features]) # save output in a list (as a back up more than anything)
                    
            for prop_i in props_which: 
                if prop_i == 'sw': prop_i_df = 'sw_perc' # we want to look at the fractional change
                else: prop_i_df = prop_i
                
                features_reject.extend(df_props[df_props['uc_'+prop_i_df] > props[prop_i][4] * unc_multiplier].index.values.tolist())
                
                # try: features_reject.extend((features_reject,df_props[df_props['uc_'+prop_i_df] > props[prop_i][4]].index.values.tolist())[1])
                # except: pass
        # labfit didn't run, sniff out the offender
        if feature_error == 'no LWA': 
            print('feature error - Labfit did not run due to unknown struggles - initiate feature sniffer')
            
            if feature_error2 == 'no LWA': 

                features_safe, features_safe_bad, features_dangerous = lab.feature_sniffer(features, d_labfit_kernal, bin_name, prop_which, props, iter_sniff=13, d_labfit_main=d_labfit_main)
                
                features_reject.extend(features_dangerous)
                features_remove.extend(features_dangerous)

                # if this is the second time in a row, also include the safe_bad features
                features_reject.extend(features_safe_bad)              
                features_remove.extend(features_safe_bad)
                
            else: 
            
                features_safe, features_safe_bad, features_dangerous = lab.feature_sniffer(features, d_labfit_kernal, bin_name, prop_which, props, d_labfit_main=d_labfit_main)
                
                features_reject.extend(features_dangerous)
                features_remove.extend(features_dangerous)
            
            feature_error2 = 'no LWA'
        
        elif feature_error2 == 'no LWA': feature_error2 = None # if things worked this round, reset feature_error2
        
        # if labfit wouldn't run, reject that feature
        if feature_error is not None and feature_error != 'no LWA': 
            print('feature error - Labfit did not run due to struggles with feature ' + str(feature_error))
            features_reject.append(feature_error) 
            features_remove.append(feature_error)
            if feature_error2 == feature_error: please = stophere # things appear to be stuck in a loop
            feature_error2 = feature_error
        
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
prop_which2 = 'n_self'
prop_which3 = 'sd_self'

# get comparative information and plot change in each parameter
# [df_compare, df_props] = lab.compare_dfs(d_labfit_kernal, d_old, bins, bin_name, props[prop_which], props_which, props[prop_which2], props[prop_which3]) # <-------------------

# plot the new spectra with old residual included
# [_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
# [T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_kernal, bins, bin_name) # <-------------------
# df_calcs = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
# lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
# plt.title(bin_name)

    
asdfsdf
  
    #%% compile the bins and remake input file

d_file = r'D:\pure water fast process large features'

df_info = pd.DataFrame()

for bin_name in os.listdir(os.path.join(d_file)): 
    
    file_name = os.listdir(os.path.join(d_file, bin_name))[-1][:-4]
    
    df_info_bin = db.labfit_to_df(os.path.join(d_file, bin_name, file_name), htp=False)
    df_info_bin = df_info_bin[df_info_bin.uc_n_self > 0]
    
    df_info = df_info.append(df_info_bin)

T = [300,500,700,900,1100,1300]
df_q = lab.information_df(d_labfit_kernal, bin_name, bins, cutoff_s296, T, df_external_load=df_info)


#%%    SD self constant (average)

df_q2 = df_q[(df_q.uc_sd_self < 0.03)&(df_q.sd_self > 0.01)&(df_q.uc_n_self < 0.03)]



for vp in df_q2.vp.unique():
    
    if len(df_q2[df_q2.vp == vp]) > 10: 
        
        plt.plot(df_q2[df_q2.vp == vp].m.abs(), df_q2[df_q2.vp == vp].sd_self, 'x')

        print('{} @ {}     meas: {}    std: {} '.format(vp, len(df_q2[df_q2.vp == vp]), np.mean(df_q2[df_q2.vp == vp].sd_self), np.std(df_q2[df_q2.vp == vp].sd_self)))
        
        plt.hlines(np.mean(df_q2[df_q2.vp == vp].sd_self), 0,15)

print('{} @ {}     meas: {}    std: {} '.format('all', len(df_q2), np.mean(df_q2.sd_self), np.std(df_q2.sd_self)))

plt.hlines(np.mean(df_q2.sd_self), 0,15, 'r')
plt.hlines(np.mean(df_q2.sd_self) + np.std(df_q2.sd_self), 0,15, 'r')
plt.hlines(np.mean(df_q2.sd_self) - np.std(df_q2.sd_self), 0,15, 'r')

plt.xlabel('|m|')
plt.ylabel('SD self')


#%%    SD self with linear fit

df_q2 = df_q[(df_q.uc_sd_self < 0.05)&(df_q.sd_self > 0.01)]


# from scipy.optimize import curve_fit

def Pade_22(m, c0, c1): 
    
    c2=0
    d2=0
    d1=0

    return c0 + c1*m

    # return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2)

for vp in df_q2.vp.unique():
    
    if len(df_q2[df_q2.vp == vp]) > 30: 
        
        plt.plot(df_q2[df_q2.vp == vp].m.abs(), df_q2[df_q2.vp == vp].sd_self, 'x')
        plt.errorbar(df_q2[df_q2.vp == vp].m.abs(), df_q2[df_q2.vp == vp].sd_self, yerr=df_q2[df_q2.vp == vp].uc_sd_self, ls='none')
        
        parameters_Pade_22, _ = curve_fit(Pade_22, df_q2[df_q2.vp == vp].m.abs(),  df_q2[df_q2.vp == vp].sd_self)
        n_self_Pade_22 = Pade_22(df_q2[df_q2.vp == vp].m.abs().sort_values(), *parameters_Pade_22)        

        plt.plot(df_q2[df_q2.vp == vp].m.abs().sort_values(), n_self_Pade_22)

parameters_Pade_22, _ = curve_fit(Pade_22, df_q2.m.abs(),  df_q2.sd_self)
n_self_Pade_22 = Pade_22(df_q2.m.abs().sort_values(), *parameters_Pade_22)        

plt.hlines(np.mean(df_q2.sd_self), 0,15, 'r')
plt.hlines(np.mean(df_q2.sd_self) + np.std(df_q2.sd_self), 0,15, 'r')
plt.hlines(np.mean(df_q2.sd_self) - np.std(df_q2.sd_self), 0,15, 'r')

plt.plot(df_q2.m.abs().sort_values(), n_self_Pade_22, 'k:', linewidth=5)

plt.xlabel('|m|')
plt.ylabel('n self')



#%%    n_self f(m)

df_q2 = df_q[(df_q.uc_n_self < 0.05)&(df_q.sd_self > 0.01)]


# from scipy.optimize import curve_fit

def Pade_22(m, c0, c1): 
    
    c2=0
    d2=0
    d1=0

    return c0 + c1*m

    # return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2)

for vp in df_q2.vp.unique():
    
    if len(df_q2[df_q2.vp == vp]) > 30: 
        
        plt.plot(df_q2[df_q2.vp == vp].m.abs(), df_q2[df_q2.vp == vp].n_self, 'x')
        plt.errorbar(df_q2[df_q2.vp == vp].m.abs(), df_q2[df_q2.vp == vp].n_self, yerr=df_q2[df_q2.vp == vp].uc_n_self, ls='none')
        
        parameters_Pade_22, _ = curve_fit(Pade_22, df_q2[df_q2.vp == vp].m.abs(),  df_q2[df_q2.vp == vp].n_self)
        n_self_Pade_22 = Pade_22(df_q2[df_q2.vp == vp].m.abs().sort_values(), *parameters_Pade_22)        

        plt.plot(df_q2[df_q2.vp == vp].m.abs().sort_values(), n_self_Pade_22)

parameters_Pade_22, _ = curve_fit(Pade_22, df_q2.m.abs(),  df_q2.n_self)
n_self_Pade_22 = Pade_22(df_q2.m.abs().sort_values(), *parameters_Pade_22)        

residuals =  Pade_22(df_q2.m.abs(), *parameters_Pade_22) - df_q2.n_self
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df_q2.n_self - np.mean(df_q2.n_self))**2)

r_squared = 1 - (ss_res / ss_tot)


plt.plot(df_q2.m.abs().sort_values(), n_self_Pade_22, 'k:', linewidth=5)

plt.xlabel('|m|')
plt.ylabel('n self')


#%%    n_self f(gamma)

# from scipy.optimize import curve_fit

df_q2 = df_q[(df_q.uc_n_self < 0.03)&(df_q.sd_self > 0.01)&(df_q.uc_sd_self < 0.03)]

def Pade_22(m, c0, c1): 
    
    c2=0
    d2=0
    d1=0

    return c0 + c1*m

    # return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2)

for vp in df_q2.vp.unique():
    
    if len(df_q2[df_q2.vp == vp]) > 30: 
        
        plt.plot(df_q2[df_q2.vp == vp].gamma_self, df_q2[df_q2.vp == vp].n_self, 'x')
        plt.errorbar(df_q2[df_q2.vp == vp].gamma_self, df_q2[df_q2.vp == vp].n_self, yerr=df_q2[df_q2.vp == vp].uc_n_self, ls='none')
        
        parameters_Pade_22, _ = curve_fit(Pade_22, df_q2[df_q2.vp == vp].gamma_self,  df_q2[df_q2.vp == vp].n_self)
        n_self_Pade_22 = Pade_22(df_q2[df_q2.vp == vp].gamma_self.sort_values(), *parameters_Pade_22)        

        plt.plot(df_q2[df_q2.vp == vp].gamma_self.sort_values(), n_self_Pade_22)

        print('{} @ {} '.format(vp, len(df_q2[df_q2.vp == vp])))
    

parameters_Pade_22, _ = curve_fit(Pade_22, df_q2.gamma_self,  df_q2.n_self)
n_self_Pade_22 = Pade_22(df_q2.gamma_self.sort_values(), *parameters_Pade_22)   

residuals =  Pade_22(df_q2.gamma_self, *parameters_Pade_22) - df_q2.n_self
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df_q2.n_self - np.mean(df_q2.n_self))**2)

r_squared = 1 - (ss_res / ss_tot)

plt.plot(df_q2.gamma_self.sort_values(), n_self_Pade_22, 'k:', linewidth=5)

plt.plot(df_q2.gamma_self.sort_values(), 0.052 + 1.53 * df_q2.gamma_self.sort_values(), 'r:', linewidth=5)


plt.xlabel('gamma self')
plt.ylabel('n self')


#%% delta_self f(m)

df_q2 = df_q[(df_q.uc_delta_self < 0.005)&(df_q.sd_self > 0.01)&(df_q.uc_delta_self > 0)]


# from scipy.optimize import curve_fit

def Pade_22(m, c0, c1): 
    
    c2=0
    d2=0
    d1=0

    return c0 + c1*m

    # return (c0 + c1*abs(m) + c2*abs(m)**2) / (1 + d1*abs(m) + d2*abs(m)**2)

for vp in df_q2.vp.unique():
    
    if len(df_q2[df_q2.vp == vp]) > 30: 
        
        plt.plot(df_q2[df_q2.vp == vp].m, df_q2[df_q2.vp == vp].delta_self, 'x')
        plt.errorbar(df_q2[df_q2.vp == vp].m, df_q2[df_q2.vp == vp].delta_self, yerr=df_q2[df_q2.vp == vp].uc_delta_self, ls='none')
        
        parameters_Pade_22, _ = curve_fit(Pade_22, df_q2[df_q2.vp == vp].m,  df_q2[df_q2.vp == vp].delta_self)
        delta_self_Pade_22 = Pade_22(df_q2[df_q2.vp == vp].m.sort_values(), *parameters_Pade_22)        

        plt.plot(df_q2[df_q2.vp == vp].m.sort_values(), delta_self_Pade_22)

parameters_Pade_22, _ = curve_fit(Pade_22, df_q2.m,  df_q2.delta_self)
delta_self_Pade_22 = Pade_22(df_q2.m.sort_values(), *parameters_Pade_22)        

residuals =  Pade_22(df_q2.m, *parameters_Pade_22) - df_q2.delta_self
ss_res = np.sum(residuals**2)
ss_tot = np.sum((df_q2.delta_self - np.mean(df_q2.delta_self))**2)

r_squared = 1 - (ss_res / ss_tot)


plt.plot(df_q2.m.sort_values(), delta_self_Pade_22, 'k:', linewidth=5)

plt.xlabel('|m|')
plt.ylabel('n self')











