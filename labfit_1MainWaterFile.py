
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
props['elower'] = ['elower', 'E\"', 4, 34, None]
props['n_air'] = ['n_air', 'n air', 5, 26, 0.13]
props['delta_air'] = ['delta_air', 'δ air', 6, 27, .005]
props['n_delta_air'] = ['n_delta_air', 'n δ air', 7, 28, 0.2]
props['MW'] = ['MW', 'MW', 8, 29, None]
props['gamma_self'] = ['gamma_self', 'γ self', 9, 30, 0.10]
props['n_self'] = ['n_self', 'n γ self', 10, 31, 0.13]
props['delta_self'] = ['delta_self', 'δ self', 11, 32, 0.005]
props['n_delta_self'] = ['n_delta_self', 'n δ self', 12, 33, 0.13]
props['beta_g_self'] = ['beta_g_self', 'βg self', 13, 35, None] # dicke narrowing (don't worry about it for water)
props['y_self'] = ['y_self', 'y self', 14, 36, None] # rosenkrantz line mixing (don't worry about this one either)
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

d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_main = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

base_name_pure = 'p2020'
d_cutoff_locations = d_labfit_main + '\\cutoff locations pure.pckl'

base_name_air = 'B2020Ja1'

if d_type == 'pure': base_name = base_name_pure
elif d_type == 'air': base_name = base_name_air

cutoff_s296 = 1E-24
ratio_min_plot = -2 # min S_max value to both plotting (there are so many tiny transitions we can't see, don't want to bog down)
offset = 2 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

# %% run specific parameters and function executions

bin_name = 'B10' # name of working bin (for these calculations)
d_old = os.path.join(d_labfit_main, bin_name, bin_name + '-000-og') # for comparing to original input files

# use_rei = True

# features_doublets_remove = []
# features_remove_manually = []

prop_which2 = False
prop_which3 = False

nudge_sd = True
features_reject_old = []

please = stophere

# %% update n parameters to match Paul (Labfit default is 0.75 for all features)

lab.nself_quantumJpp(d_labfit_main, base_name_pure)

lab.nself_quantumJpp(d_labfit_main, 'B10')


base_name = base_name + 'n'

# %% mini-script to get the bin started, save OG file

lab.bin_ASC_cutoff(d_labfit_main, base_name, d_labfit_main, bins, bin_name, d_cutoff_locations, d_conditions)
lab.run_labfit(d_labfit_main, bin_name) # <-------------------

lab.save_file(d_labfit_main, bin_name, d_og=True) # make a folder for saving and save the original file for later


# %% start list of things to float, plot stuff

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_main, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
a_features_check = [int(x) for x in list(df_calcs[df_calcs.ratio_max>0].index)]
# print(a_features_check)

lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = a_features_check, axis_labels=False) # <-------------------
plt.title(bin_name)

# print(a_features_check)

# lab.plot_spectra(T,wvn,trans,res,False, False, offset, features = False) # don't plot the feature names

# lab.run_labfit(d_labfit_main, bin_name, use_rei=True) # run REI file in labfit (DOES NOT SAVE INP)


# %% add new features (if required)

features_new = [6718.18, 6739.80, 6740.19] 

lab.add_features(d_labfit_main, bin_name, features_new, use_which='rei_saved') 
lab.run_labfit(d_labfit_main, bin_name) # <------------------

# lab.save_file(d_labfit_main, bin_name, d_save_name)


#%% figure out how much to shrink features that you can't see

features_shrink = [6855]

print(lab.shrink_feature(df_calcs[df_calcs.index.isin(features_shrink)], cutoff_s296, T))


#%% zoom in on some features

a = df_calcs[df_calcs.index.isin([7009,7011])]

# %% make changes to features that don't look great to begin with


lab.float_lines(d_labfit_main, bin_name, features_sw, props['sw'], 'rei_saved', features_constrain) # float lines, most recent saved REI in -> INP out
lab.float_lines(d_labfit_main, bin_name, features_nu, props['nu'], 'inp_new', features_constrain) # INP -> INP, testing two at once (typically nu or n_self)


lab.run_labfit(d_labfit_main, bin_name) # <------------------

[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_main, bin_name, bins, cutoff_s296, T, d_old) # <-------------------
a_features_check = [int(x) for x in list(df_calcs[df_calcs.ratio_max>0].index)]
lab.plot_spectra(T,wvn,trans,res,False, df_calcs[df_calcs.ratio_max>ratio_min_plot], offset, features = a_features_check, labels=False) # <-------------------
plt.title(bin_name)

# lab.save_file(d_labfit_main, bin_name, d_save_name)


# %% mini-script to check fits for specified features (will need to snag prop_which and feature parameters from txt file)


features_remove = [] # features that were already floated but don't meet uncertianty requirements
df_iter = {}
feature_error = None
feature_error2 = None

for [prop_which, prop_which2, prop_which3, d_save_name] in prop_whiches: # run through a bunch of senarios

    if d_save_name == 'only sw': # use all rejected features
        features_test = a_features_reject.copy()
        features_doublets = a_features_doublets_reject.copy()
            
    elif (d_save_name == 'sw cleanup after only nu after only sw') or (d_save_name == 'after n delta self' and len(prop_whiches) > 1): # use all accepted features
        features_test = a_features.copy()
        features_doublets = a_features_constrain.copy()

    # otherwise pass: float the input features or the same ones as last iteration

    features = features_test.copy()
    features.sort()
    
    features_constrain = features_doublets.copy()
    
    iters = 10 # default number of times to iterate 
    features_doublets_reject = []; feature_error2 = None # reset error checkers
    features_reject = [0] # make sure we get in the while loop
    
    df_iter[d_save_name] = [[features_test.copy(), features_doublets.copy()]]; 
    
    while len(features_reject) > 0 or feature_error is not None: # until an iteration doesn't reject any features (or feature_error repeats itself)
    
        features_reject = []; feature_error = None # reset these guys for this round of testing
     
        lab.float_lines(d_labfit_main, bin_name, features, props[prop_which], 'rei_saved', features_constrain) # float lines, most recent saved REI in -> INP out
        if prop_which2 is not False: lab.float_lines(d_labfit_main, bin_name, features, props[prop_which2], 'inp_new', features_constrain) # INP -> INP, testing two at once (typically nu or n_self)
        if prop_which3 is not False: lab.float_lines(d_labfit_main, bin_name, features, props[prop_which3], 'inp_new', features_constrain, nudge_sd) # INP -> INP, testing two at once (typically sd_self)
        
        print('     labfit iteration #1')
        feature_error = lab.run_labfit(d_labfit_main, bin_name) # need to run one time to send INP info -> REI
        
        i = 1 # start at 1 because we already ran things once
        while feature_error is None and i < iters: # run 10 times
            i += 1
            print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
            feature_error = lab.run_labfit(d_labfit_main, bin_name, use_rei=True) 
    
        if feature_error is None: # if we made if through all iterations without a feature causing an error...
    
            [df_compare, df_props] = lab.compare_dfs(d_labfit_main, d_old, bins, bin_name, props[prop_which], props_which, props[prop_which2], plots = False) # read results into python
            # df_iter[d_save_name].append([df_compare, df_props, features]) # save output in a list (as a back up more than anything)
                    
            for prop_i in props_which: 
                if prop_i == 'sw': prop_i_df = 'sw_perc' # we want to look at the fractional change
                else: prop_i_df = prop_i
                
                features_reject.extend(df_props[df_props['uc_'+prop_i_df] > props[prop_i][4]].index.values.tolist())
                
                # try: features_reject.extend((features_reject,df_props[df_props['uc_'+prop_i_df] > props[prop_i][4]].index.values.tolist())[1])
                # except: pass
            
        # if labfit wouldn't run, reject that feature
        if feature_error is not None: 
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
        # if you're not floating anything, don't bother looping through things as intensely
        if features == []: iters = 0 
        
    a_features_reject = sorted(list(set(features_test).difference(set(features)))) # watch features that were removed  
    a_features = sorted(features) # list in a list for copying to notepad
    a_features_constrain = sorted(features_constrain)
    a_features_doublets_reject = sorted(features_doublets_reject)
    a_features_remove = sorted(features_remove)
    
    df_iter[d_save_name].append([a_features.copy(), a_features_constrain.copy(), a_features_reject.copy(), a_features_doublets_reject.copy(), a_features_remove.copy()])
                
    lab.save_file(d_labfit_main, bin_name, d_save_name) # save file, this is what you will be reloading for the next round of the for loop  

print(' *** these features were manually removed - you might need to remove them in the save rei file *** ')
print(features_remove_manually)
a_features_remove_manually = features_remove_manually
        
if d_save_name == 'sw cleanup after only nu after only sw': prop_which2 = 'nu'
elif d_save_name == 'after gamma_self - take what we can get': prop_which2 = 'n_self'; prop_which
elif d_save_name == 'after n delta self': prop_which2 = 'delta_self'

[df_compare, df_props] = lab.compare_dfs(d_labfit_main, d_old, bins, bin_name, props[prop_which], props_which, props[prop_which2], props[prop_which3]) # <-------------------

[_, _,   _,     _, res_og,      _,     _,           _] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name, og=True) # <-------------------
[T, P, wvn, trans, res, wvn_range, cheby, zero_offset] = lab.labfit_to_spectra(d_labfit_main, bins, bin_name) # <-------------------
df_calcs = lab.information_df(d_labfit_main, bin_name, bins, cutoff_s296, T, d_old=d_old) # <-------------------
lab.plot_spectra(T,wvn,trans,res,res_og, df_calcs, offset, props[prop_which], props[prop_which2], axis_labels=False) # <-------------------
plt.title(bin_name)




# df_props[df_props.index == 23683]

# lab.save_file(d_labfit_main, bin_name, 'fixed quantum Jpp for n_self relationship and removed 6988 float')



#%% feature sniffer (find which feature is the problem)

features_test = features_test = [16848, 16860, 16903, 16908, 16916, 16987, 17000, 17051, 17064, 17120, 17125, 17142, 17158, 17394, 17483, 17564, 17610]
features_doublet = []

features_safe = []
features_dangerous = []
iter_sniff = 2

for feature in features_test: # sniff out which feature(s) are causing you grief
    print(feature)
    lab.float_lines(d_labfit, bin_name, [feature], props[prop_which], use_saved, []) # float lines, most recent saved REI in -> INP out
    if prop_which2 is not False: lab.float_lines(d_labfit, bin_name, [feature], props[prop_which2], use_inp, []) # INP -> INP, testing two at once (typically nu or n_self)
    if prop_which3 is not False: lab.float_lines(d_labfit, bin_name, [feature], props[prop_which3], use_inp, [], nudge_sd) # INP -> INP, testing two at once (typically sd_self)
    try: 
        feature_error = lab.run_labfit(d_labfit, bin_name) # need to run one time to send INP info -> REI
        i = 1 # start at 1 because we already ran things once
        while feature_error is None and i < iter_sniff: # run x number of times
            i += 1
            print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
            feature_error = lab.run_labfit(d_labfit, bin_name, use_rei) 
        if feature_error is None: 
          features_safe.append(feature)
        else: 
            feature_error = None
            throw = thaterrorplease
    except: 
        features_dangerous.append(feature)

for doublet in features_doublets: # delete doublets from the features_test list (they'll all be run here)
    lab.float_lines(d_labfit, bin_name, doublet, props[prop_which], use_saved, [doublet])
    if prop_which2 is not False: lab.float_lines(d_labfit, bin_name, doublet, props[prop_which2], use_inp, [doublet]) # INP -> INP, testing two at once (typically nu or n_self)
    if prop_which3 is not False: lab.float_lines(d_labfit, bin_name, doublet, props[prop_which3], use_inp, [doublet], nudge_sd) # INP -> INP, testing two at once (typically sd_self)
    try: 
        feature_error = lab.run_labfit(d_labfit, bin_name) # need to run one time to send INP info -> REI
        i = 1 # start at 1 because we already ran things once
        while feature_error is None and i < iter_sniff: # run x number of times
            i += 1
            print('     labfit iteration #' + str(i)) # +1 for starting at 0, +1 again for already having run it using the INP (to lock in floats)
            feature_error = lab.run_labfit(d_labfit, bin_name, use_rei) 
        if feature_error is None: 
          features_safe.append(doublet)
        else: 
            feature_error = None
            throw = thaterrorplease
    except: 
        features_dangerous.append(doublet)




