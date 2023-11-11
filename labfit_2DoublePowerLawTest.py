r'''

labfit2 double power law testing

process measurements one temperature at a time and then compile and compare


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


please = stophere

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


r'''


#%% fit the data for n values

from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

def SPL(T, c, n): 
    
    return c*(296/T)**n

def SPLoff(T, c1, c2, n): 

    return c1*(296/T)**n + c2
    
def DPL(T, c1, n1, c2, n2): 
    
    return c1*(296/T)**n1 + c2*(296/T)**n2


#%% load in the DPL data

f = open(os.path.join(d_sceg_save,'DPL exploration.pckl'), 'rb')
[output_dpl, output_lab, T_conditions, features_strong, features_doublets] = pickle.load(f)
f.close()     

T_conditions = [float(T) for T in T_conditions]
T_conditions = np.asarray(T_conditions)

features_strong_flat = [item for sublist in features_strong for item in sublist]

for doublet in features_doublets: 
    if doublet != []: 
        for sub_doublet in doublet: 
            features_strong_flat.remove(sub_doublet[1])

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

output_type = output_dpl[i_type,:,:,:]

plt.figure()
i = 0

for i_feat, feat in enumerate(features_strong_flat): 
    
    prop = output_type[i_feat,:,i_prop]
    uc_prop = output_type[i_feat,:,i_prop+1]
    
    if (-1 not in uc_prop) and (max(uc_prop)<0.2): # and (min(prop>0.01)): 
        
        T_plot = [T+i_feat/10 for T in T_conditions]
        
        plt.plot(T_plot, prop)
        plt.errorbar(T_plot, prop, yerr=uc_prop)
        
        if i == 0: 
            prop_all = prop.copy()
            i+=1
        else: 
            prop_all = np.vstack((prop_all,prop))
            
            
prop_median = np.median(prop_all, axis=0)

prop_mean = np.mean(prop_all, axis=0)
prop_perc = 100*(prop_mean-np.mean(prop_mean)) / np.mean(prop_mean)
prop_perc_std = np.std(prop_perc)

plt.plot(T_conditions, prop_mean, 'k', linewidth=10, zorder=100, label='mean (T)')
plt.plot(T_conditions, prop_median, 'r', linewidth=10, zorder=100, label='median (T)')


# SPL(T, c, n)
fit_x = T_conditions.copy()
fit_y = prop_median.copy()

    
n = 1

if prop_name == 'sd' and type_name == 'air': 
    c1 = 0.1
    c2 = 0.1
    
    # fit the data using the function  
    fit_params, _ = curve_fit(SPLoff, fit_x, fit_y, p0=[c1,c2,n])

    c1, c2, n = fit_params

    prop_fit = SPLoff(fit_x, c1, c2, n)
    
else: 
    c = 0.1
    
    # fit the data using the function  
    fit_params, _ = curve_fit(SPL, fit_x, fit_y, p0=[c,n])

    c, n = fit_params

    prop_fit = SPL(fit_x, c, n)




plt.plot(fit_x, prop_fit, 'X', color='darkgreen', markersize=10, linewidth=10, zorder=101, label='SPL fit (to median)')


mad = np.mean(np.abs(fit_y-prop_fit))
rms = np.sqrt(np.sum((fit_y-prop_fit)**2)/ len(fit_y))
r2 = r2_score(fit_y, prop_fit)


print('n={:.3f}'.format(n))
print('{}    {}     {}'.format(mad, rms, r2))

plt.legend()

plt.xlabel('Temperature (K)')
plt.ylabel('{}_{}'.format(prop_name, type_name))

print('{}_{}  {:.3f} +/- {:.2f}% for {} features\n\n\n'.format(prop_name, type_name, np.mean(prop_mean), prop_perc_std, np.shape(prop_all)[0]))





