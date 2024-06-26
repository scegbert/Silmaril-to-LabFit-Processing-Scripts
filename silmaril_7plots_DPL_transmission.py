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

import td_support as td

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

from copy import deepcopy

import pickle

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# %% define some dictionaries and parameters

d_sceg_save = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

wvn2_data = [6600, 7600] # max span of the data

#%% load in the DPL data


f = open(os.path.join(d_sceg_save,'DPL results.pckl'), 'rb')
[df_sceg, T_conditions, features_strong] = pickle.load(f)
f.close()     

T_conditions = [float(T) for T in T_conditions]
T_conditions = np.asarray(T_conditions)




# ditch the number of iterations column
df_sceg = df_sceg.loc[:,~df_sceg.columns.str.startswith('iter_')]
df_sceg_ex = pd.read_excel(os.path.join(d_sceg_save,'DPL results.xlsx'), index_col=0)

# replace values that were updated manually, constrained doublets are now rows of NAN (not in excel file)
df_sceg.iloc[:, 48:] = df_sceg_ex.iloc[:, 48:]


f = open(os.path.join(d_sceg_save,'df_sceg_air.pckl'), 'rb')
[df_calcs, _, _, _, _, _] = pickle.load(f)
f.close()



#%% set up fit equations

def SPL(T, c, n): 
    
    return c*(296/T)**n

def SPLoff(T, c1, c2, n): 

    return c1*(296/T)**n + c2
    
def DPL(T, c1, n1, c2, n2): 
    
    return c1*(296/T)**n1 + c2*(296/T)**n2


#%% load pure H2O transmission data

f = open(os.path.join(d_sceg_save,'spectra_pure.pckl'), 'rb')
[T_pure, P_pure, wvn_pure_raw, trans_pure_raw, res_pure_raw, res_HT_pure_raw, res_sd0_raw] = pickle.load(f)
f.close()

[T_all_pure, P_all_pure] = np.asarray([T_pure, P_pure])

which_files_pure = ['300 K 16 T', '500 K 16 T', '700 K 16 T', '900 K 16 T', '1100 K 16 T', '1300 K 16 T']

res_HT_pure = []
res_updated_pure = []
trans_pure = []
wvn_pure = []

for which_file in which_files_pure: 

    T_plot = int(which_file.split()[0])
    P_plot = int(which_file.split()[2])
    
    i_plot = np.where((T_all_pure == T_plot) & (P_all_pure == P_plot))[0]
        
    T = [T_all_pure[i] for i in i_plot]
    P = [P_all_pure[i] for i in i_plot]
    
    wvn_labfit_all = np.concatenate([wvn_pure_raw[i] for i in i_plot])
    trans_labfit_all = np.concatenate([trans_pure_raw[i] for i in i_plot])
    res_updated_all = np.concatenate([res_pure_raw[i] for i in i_plot])
    res_HT_all = np.concatenate([res_HT_pure_raw[i] for i in i_plot])
    
    
    [istart, istop] = td.bandwidth_select_td(wvn_labfit_all, wvn2_data, max_prime_factor=500, print_value=False)
    
    wvn_pure.append(wvn_labfit_all[istart:istop])
    trans_pure.append(trans_labfit_all[istart:istop]/100)
    res_updated_pure.append(res_updated_all[istart:istop]/100)
    res_HT_pure.append(res_HT_all[istart:istop]/100) 

#%% load air H2O transmission data

f = open(os.path.join(d_sceg_save,'spectra_air.pckl'), 'rb')
[T_air, P_air, wvn_air_raw, trans_air_raw, res_air_raw, res_HT_air_raw, res_sd0_raw] = pickle.load(f)
f.close()

[T_all_air, P_all_air] = np.asarray([T_air, P_air])
P_all_air = np.round(P_all_air/10,0)*10 # for air-water to round pressures to nearest 10's

which_files_air = ['300 K 600 T', '500 K 600 T', '700 K 600 T', '900 K 600 T', '1100 K 600 T', '1300 K 600 T']

res_HT_air = []
res_updated_air = []
trans_air = []
wvn_air = []

for which_file in which_files_air: 

    T_plot = int(which_file.split()[0])
    P_plot = int(which_file.split()[2])
    
    i_plot = np.where((T_all_air == T_plot) & (P_all_air == P_plot))[0]
        
    T = [T_all_air[i] for i in i_plot]
    P = [P_all_air[i] for i in i_plot]
    
    wvn_labfit_all = np.concatenate([wvn_air_raw[i] for i in i_plot])
    trans_labfit_all = np.concatenate([trans_air_raw[i] for i in i_plot])
    res_updated_all = np.concatenate([res_air_raw[i] for i in i_plot])
    res_HT_all = np.concatenate([res_HT_air_raw[i] for i in i_plot])
    
    
    [istart, istop] = td.bandwidth_select_td(wvn_labfit_all, wvn2_data, max_prime_factor=500, print_value=False)
    
    wvn_air.append(wvn_labfit_all[istart:istop])
    trans_air.append(trans_labfit_all[istart:istop]/100)
    res_updated_air.append(res_updated_all[istart:istop]/100)
    res_HT_air.append(res_HT_all[istart:istop]/100) 



#%% HITRAN uncertainties

HT_uncertainty = {'0': False,  # '0 (unreported)', 
                  '1': False,  # '1 (default)', 
                  '2': False,  # '2 (average)', 
                  '3': False, # '3 (over 20%)', 
                  '4': 0.20, # '4 (10-20%)', 
                  '5': 0.10, # '5 (5-10%)', 
                  '6': 0.05, # '6 (2-5%)', 
                  '7': 0.02, # '7 (1-2%)', 
                  '8': 0.01} # '8 (under 1%)'}


T_unc_pure = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 19/1288])
T_unc_air = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 20/1288])

P_unc_pure = 0.0027
P_unc_air = 0.0025
y_unc_air = 0.029

T_smooth = np.linspace(min(T_all_air)-200, max(T_all_air)+200, 5000)

please = stop_here____fully_loaded

#%% DPL plots with transmission for Parts I and II (only for air or pure, not both at the same time)

d_type = 'pure' # 'pure' or 'air'

partials = True
aw = True # plot aw (if false, plot SD as gamma_2)
equations_in_legend = False

if d_type == 'pure': 

    features_plot = [14817, 12952]
    nu_span =  [0.06, 0.08]
    y_span0 = [0.76, 0.21]
    y_span1 = [[-0.049, 0.02], [-0.075, 0.031]]
    
    which_files = which_files_pure.copy()
    wvn = wvn_pure.copy()
    trans = trans_pure.copy()
    res_updated = res_updated_pure.copy()
    res_HT = res_HT_pure.copy()
    
    nu_spacing = 0.08

elif d_type == 'air':
    
    features_plot = [17112, 27034,  13950]
    nu_span =  [0.35, 0.55, 0.16]
    y_span0 = [0.901, .301, 0.8601]
    y_span1 = [[-0.0105, 0.003], [-0.034, 0.022], [-0.067, 0.057]]
    
    which_files = which_files_air.copy()
    wvn = wvn_air.copy()
    trans = trans_air.copy()
    res_updated = res_updated_air.copy()
    res_HT = res_HT_air.copy()

    nu_spacing = 0.18


# features_plot = [16467, 17300, 17423, 17955, 18406, 18555, 19055, 19406,
#                  20349, 20429, 20835, 21484, 21873, 21929, 22422, 22431,
#                  22455, 23200, 23360, 23615, 24324, 24605, 25695, 26578,
#                  27622, 27730, 28543, 30781, 31555, 32453, 32958, 33330,
#                  33347, 33603, 33706, 33757, 33811, 34111, 34360, 34617,
#                  34834, 34962, 35005, 35251, 35597]

# features_plot = [17300, 18406, 19055, 32958, 33706, 34617, 35005, 35251]

colors_fits = ['lime','','indigo', 'darkorange']



if d_type == 'pure': 
    name_plot = ['Self-Width\nγ$_{self}$ [cm$^{-1}$/atm]', 
                 'Self-Shift\nδ$_{self}$ [cm$^{-1}$/atm]', 
                 'Speed Dependence\nof the Self-Width, a$_{w}$']
    
    data_HT = [['gamma_self',False], 
               False, 
               False]

    uc_HT = [[3,False],
              [False],
              [False]]

    data_labfit = ['n_self', 
                   'n_delta_self', 
                   'sd_self']


elif d_type == 'air': 
    name_plot = ['Air-Width\nγ$_{air}$ [cm$^{-1}$/atm]', 
                 'Air-Shift\nδ$_{air}$ [cm$^{-1}$/atm]', 
                 'Speed Dependence\nof the Air-Width, a$_{w}$']

    data_HT = [['gamma_air', 'n_air'], 
               ['delta_air', False], 
               False]

    uc_HT = [[2,4],
              [5, False],
              [False]]
    
    data_labfit = ['n_air', 
                   'n_delta_air', 
                   'sd_air']

iter_g = 0
iter_d = 0
iter_sd = 0

df_unc = pd.DataFrame(index=df_sceg.index, columns=['uc_gamma_max','uc_delta_max','uc_sd_max','sd_min','sd_min_unc'])

df_sceg_updated_uc = df_sceg.copy()

for i_feat, feat in enumerate(features_plot): # enumerate([32958]): # enumerate(df_sceg.index): # 
    
    feat = int(feat)
    
    df_feat = df_sceg.loc[feat]   
    

    if d_type == 'pure': 
        prop_index = [index for index in df_feat.index if '_self_' in index]
        iter_index = [index for index in df_feat.index if 'iter_self_' in index]
        sd_index = [index for index in df_feat.index if 'sd_self_' == index[:8]]
        
        uc_g_index = [index for index in df_feat.index if 'uc_gamma_self_' in index]
        uc_d_index = [index for index in df_feat.index if 'uc_delta_self_' in index]
        uc_sd_index = [index for index in df_feat.index if 'uc_sd_self_' in index]
        
        all_g_index = [index for index in df_feat.index if 'gamma_self_' in index]
        all_d_index = [index for index in df_feat.index if 'delta_self_' in index]
        all_sd_index = [index for index in df_feat.index if 'sd_self_' in index]
        
    elif d_type == 'air': 
        prop_index = [index for index in df_feat.index if '_air_' in index]
        iter_index = [index for index in df_feat.index if 'iter_air_' in index]
        sd_index = [index for index in df_feat.index if 'sd_air_' == index[:7]]
        
        uc_g_index = [index for index in df_feat.index if 'uc_gamma_air_' in index]
        uc_d_index = [index for index in df_feat.index if 'uc_delta_air_' in index]
        uc_sd_index = [index for index in df_feat.index if 'uc_sd_air_' in index]
        
        all_g_index = [index for index in df_feat.index if 'gamma_air_' in index]
        all_d_index = [index for index in df_feat.index if 'delta_air_' in index]
        all_sd_index = [index for index in df_feat.index if 'sd_air_' in index]
        
        

       
    df_unc.loc[feat,'uc_gamma_max'] = max(df_feat[uc_g_index])
    df_unc.loc[feat,'uc_delta_max'] = max(df_feat[uc_d_index])
    df_unc.loc[feat,'uc_sd_max'] = max(df_feat[uc_sd_index])

    df_unc.loc[feat,'sd_min'] = min(df_feat[sd_index])
    df_unc.loc[feat,'sd_min_unc'] = min(df_feat[sd_index] - df_feat[uc_sd_index].to_numpy())
    
    which = (~np.any(df_feat[prop_index].isna())) # no NAN's
             
    if which:

        df_gamma_self = df_feat[df_feat.index.str.startswith('gamma_self_')]
        df_uc_gamma_self = df_feat[df_feat.index.str.startswith('uc_gamma_self_')]
        df_uc_gamma_self[:] = np.sqrt((df_uc_gamma_self.to_numpy(float)/df_gamma_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_gamma_self
        
        df_sd_self = df_feat[df_feat.index.str.startswith('sd_self_')]
        df_uc_sd_self = df_feat[df_feat.index.str.startswith('uc_sd_self_')]
        df_uc_sd_self[:] = np.sqrt((df_uc_sd_self.to_numpy(float)/df_sd_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_sd_self # droped 3.9% due to TD of SD
        
        df_delta_self = df_feat[df_feat.index.str.startswith('delta_self_')]
        df_uc_delta_self = df_feat[df_feat.index.str.startswith('uc_delta_self_')]
        df_uc_delta_self[:] = np.sqrt((df_uc_delta_self.to_numpy(float)/df_delta_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2 + 
                                      (1.7E-4 / (0.021*df_delta_self.to_numpy(float)))**2) * abs(df_delta_self)
                                      
        
        df_gamma_air = df_feat[df_feat.index.str.startswith('gamma_air_')]
        df_uc_gamma_air = df_feat[df_feat.index.str.startswith('uc_gamma_air_')]
        df_uc_gamma_air[:] = np.sqrt((df_uc_gamma_air.to_numpy(float)/df_gamma_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_gamma_air
                
        df_sd_air = df_feat[df_feat.index.str.startswith('sd_air_')]
        df_uc_sd_air = df_feat[df_feat.index.str.startswith('uc_sd_air_')]
        df_uc_sd_air[:] = np.sqrt((df_uc_sd_air.to_numpy(float)/df_sd_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_sd_air # droped 5.2
        
        df_delta_air = df_feat[df_feat.index.str.startswith('delta_air_')]
        df_uc_delta_air = df_feat[df_feat.index.str.startswith('uc_delta_air_')]
        df_uc_delta_air[:] = np.sqrt((df_uc_delta_air.to_numpy(float)/df_delta_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 +  (y_unc_air)**2 +
                                      (1.7E-4 / (0.789*df_delta_air.to_numpy(float)))**2) * abs(df_delta_air)
        
        
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_gamma_self_')] = df_uc_gamma_self
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_sd_self_')] = df_uc_sd_self
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_delta_self_')] = df_uc_delta_self
        
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_gamma_air_')] = df_uc_gamma_air
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_sd_air_')] = df_uc_sd_air
        df_sceg_updated_uc.loc[feat,df_feat.index.str.startswith('uc_delta_air_')] = df_uc_delta_air
        
        
        fig = plt.figure(figsize=(6.5*2, 3.6*2)) 
        gs = GridSpec(4, 2, figure=fig, width_ratios=[2,1], height_ratios=[2,2,1,1], hspace = 0, wspace = 0.02) 
        axs_data = []
        axs_data.append(fig.add_subplot(gs[0,0]))
        axs_data.append(fig.add_subplot(gs[1,0], sharex=axs_data[0]))
        axs_data.append(fig.add_subplot(gs[2:,0], sharex=axs_data[0]))
        
        
        axs_trans = []
        axs_trans.append(fig.add_subplot(gs[:2,1]))
        axs_trans.append(fig.add_subplot(gs[2,1], sharex=axs_trans[0]))
        axs_trans.append(fig.add_subplot(gs[3,1], sharex=axs_trans[0]))
                
        
        
        
        quanta = df_feat.quanta.split()
        try: kcJ = float(quanta[11])/float(quanta[9])
        except: kcJ = 0.
               
        title = '{}{}{} ← {}{}{}      {}$_{}$$_{}$ ← {}$_{}$$_{}$      Kc" / J" = {:.3f}'.format(quanta[0],quanta[1],quanta[2],quanta[3],quanta[4],quanta[5],
                                                                        quanta[6],quanta[7],quanta[8],quanta[9],quanta[10],quanta[11], kcJ)
        
        # plt.suptitle(title)
        
        
        if d_type == 'pure': 
            if aw: sd_mult = 1
            else: sd_mult =df_gamma_self.to_numpy()
            
            data_plots = [[df_gamma_self, df_uc_gamma_self], 
                          [df_delta_self, df_uc_delta_self], 
                          [df_sd_self *sd_mult, df_uc_sd_self *sd_mult]]
            g_unc = 0.10
            
        elif d_type == 'air': 
            if aw: sd_mult = 1
            else: sd_mult =df_gamma_air.to_numpy()
            
            data_plots = [[df_gamma_air, df_uc_gamma_air], 
                          [df_delta_air, df_uc_delta_air], 
                          [df_sd_air *sd_mult, df_uc_sd_air *sd_mult]]
            
            g_unc = 0.012
            
        d_unc = 0.005
        sd_unc = 0.1
        mult_unc = 1
        
        sd_cutoff = 0.05

        for i_plot, [data, uc_data] in enumerate(data_plots): 

            
            if partials: 
                if data.index[0].split('_')[0] == 'gamma': 
                    which = (~np.any(df_feat[uc_g_index]>g_unc*mult_unc)) # check if uncertainies below threshold
                    if which: 
                        iter_g+=1
                    else: 
                        df_sceg.loc[feat,all_g_index] = np.nan
                        
                elif data.index[0].split('_')[0] == 'delta': 
                    which = (~np.any(df_feat[uc_d_index]>d_unc*mult_unc)) # check if uncertainies below threshold
                    if which: 
                        iter_d+=1
                    else: 
                        df_sceg.loc[feat,all_d_index] = np.nan
                    
                elif data.index[0].split('_')[0] == 'sd': 
                                        
                    which = ((~np.any(df_feat[uc_sd_index]>sd_unc*mult_unc))&
                             (df_unc.loc[feat,'sd_min_unc']>0.0)) # check if uncertainies below threshold, value larger than 0
                    if which: 
                        iter_sd+=1
                    else: 
                        df_sceg.loc[feat,all_sd_index] = np.nan
                        
            else: 
                
                which = ((~np.any(df_feat[uc_g_index]>g_unc*mult_unc))&
                         (~np.any(df_feat[uc_d_index]>d_unc*mult_unc))&
                         (~np.any(df_feat[uc_sd_index]>sd_unc*mult_unc))&
                                  (df_unc.loc[feat,'sd_min_unc']>0.0)) # check if uncertainies below threshold
                    
                if which: 
                    iter_g+=1
                    iter_d+=1
                    iter_sd+=1
           
            if True:  # which: 
                                
                axs_data[i_plot].plot(T_conditions, data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
                axs_data[i_plot].errorbar(T_conditions, data, uc_data, color='k', fmt='none', capsize=5, zorder=10)
    
                    
                # overlay HITRAN prediction
                if data_HT[i_plot]: 
                    
                    base = df_feat[data_HT[i_plot][0] + 'HT']
                    if data_HT[i_plot][1]:
                        n = df_feat[data_HT[i_plot][1] + 'HT']
                    else: 
                        n=0
                    
                    if uc_HT[i_plot]: 
                        uc_base_str = df_feat.ierr[uc_HT[i_plot][0]]
                        
                        if HT_uncertainty[uc_base_str]: 
                            uc_base = base * HT_uncertainty[uc_base_str]
                        else: uc_base = 0
                        
                        if uc_HT[i_plot][1]: 
                            uc_n_str = df_feat.ierr[uc_HT[i_plot][1]]
                            
                            if HT_uncertainty[uc_n_str]: 
                                uc_n = n * HT_uncertainty[uc_n_str]
                            else: uc_n = 0
                        else: uc_n = 0
                            
                    y_center = SPL(T_smooth, base, n)
                    y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                    SPL(T_smooth, base+uc_base, n-uc_n), 
                                    SPL(T_smooth, base-uc_base, n+uc_n),
                                    SPL(T_smooth, base-uc_base, n-uc_n)])
                    y_max = np.max(y_unc,axis=0)
                    y_min = np.min(y_unc,axis=0)
                    
                    if equations_in_legend: 
                        
                        if data_HT[i_plot][0] == 'gamma_self': 
                            axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}'.format(base))
                        elif data_HT[i_plot][0] == 'gamma_air': 
                            axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                        elif data_HT[i_plot][0] == 'delta_air': 
                            if uc_base_str in ['0','1','2']: 
                                axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f} (no TD, no uncertainty listed)'.format(base))
                            else: 
                                axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f} (no TD)'.format(base))
                        axs_data[i_plot].fill_between(T_smooth, y_min, y_max, color=colors_fits[3], alpha=.2)
                    else: 
                        axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN')

                
                # overlay Labfit prediction (if there is one)
                if df_feat['uc_'+data.index[0][:-4]] != -1: 
                    
                    color_labfit = colors_fits[0]
                    
                    if data_labfit[i_plot][:2] == 'sd':  
                        
                        sd = df_feat[data_labfit[i_plot]]
                        
                        if aw: 
                            
                            y_center = sd * np.ones_like(T_smooth)
                            y_max = y_center + df_feat['uc_'+data_labfit[i_plot]]
                            y_min = y_center - df_feat['uc_'+data_labfit[i_plot]]
                            
                        else: 
                        
                            base = df_feat[data_labfit[i_plot]] * df_feat['gamma_'+data_labfit[i_plot].split('_')[-1]]
                            n = df_feat['n_'+data_labfit[i_plot].split('_')[-1]]
                            
                            y_center = SPL(T_smooth, base, n)
                            y_max = y_center # + df_feat['uc_'+data_labfit[i_plot]]
                            y_min = y_center # - df_feat['uc_'+data_labfit[i_plot]]
                        
                    else:                       
                        
                        base = df_feat[data.index[0][:-4]]
                        uc_base = df_feat['uc_'+data.index[0][:-4]]
                        
                        n = df_feat[data_labfit[i_plot]]
                        uc_n = df_feat['uc_'+data_labfit[i_plot]]
                        if uc_n == -1: 
                            uc_n = 0  
                            # color_labfit = 'darkgreen'
                        
                        y_center = SPL(T_smooth, base, n)
                        y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                        SPL(T_smooth, base+uc_base, n-uc_n), 
                                        SPL(T_smooth, base-uc_base, n+uc_n),
                                        SPL(T_smooth, base-uc_base, n-uc_n)])
                        y_max = np.max(y_unc,axis=0)
                        y_min = np.min(y_unc,axis=0)
                                 
                    if equations_in_legend:
                    
                        if data_labfit[i_plot][:2] == 'sd': 
                            axs_data[i_plot].plot(T_smooth, y_center, color=color_labfit, label='Multispectrum Fit TW a$_w$ = {:.2f}'.format(sd), 
                                                  linestyle='dashed')
                        else: 
                            axs_data[i_plot].plot(T_smooth, y_center, color=color_labfit, label='Multispectrum Fit TW {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n), 
                                                  linestyle='dashed')
                    else: 
                        axs_data[i_plot].plot(T_smooth, y_center, color=color_labfit, label='Multispectrum Fit TW', linestyle='dashed')

                    axs_data[i_plot].fill_between(T_smooth, y_min, y_max, color=color_labfit, alpha=.2)
                
                if data_labfit[i_plot].split('_')[1] == 'delta': 
                    axs_data[i_plot].plot(T_smooth,np.zeros_like(T_smooth), color='k', linestyle=':')
                                
                if data_labfit[i_plot][:2] != 'sd': 
        
                    # fit the data using DPL (weighted by Labfit uncertainties)
                    solving=True
                    i_solve=-1
                    
                    if df_feat['uc_' + data_labfit[i_plot]] == -1: n = 0.75
                    
                    p0s = [[2*data[0], n, -2*data[0], -1*n], [data[0], n, -1*data[0], -1*n], [data[0]/2, n, data[0]/2, -1*n]]
                    while solving:
                        i_solve+=1
                        try: 
                            p_DPL, cov = fit(DPL, T_conditions, data, p0=p0s[i_solve], maxfev=5000, sigma=uc_data.to_numpy(float))
                            p_err = np.sqrt(np.diag(cov))
        
                            solving=False # you solved it!
                            
                            if equations_in_legend:
                            
                                if p_DPL[2] > 0: sign = '+'
                                else: sign = ''
                                axs_data[i_plot].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], 
                                                  label='DPL Fit TW {:.3f}(T/T)$^{{{:.2f}}}${}{:.3f}(T/T)$^{{{:.2f}}}$'.format(p_DPL[0],p_DPL[1],sign,p_DPL[2],p_DPL[3]), 
                                                                         linestyle='dashdot')
                            else:
                                axs_data[i_plot].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], 
                                                  label='DPL Fit TW', linestyle='dashdot')
                                
                            
                            # calculate uncertainties (too many Infs and NANS - haven't been using much)
                            y_unc = np.array([DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                              DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                              DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3])])
                            y_max = np.max(y_unc,axis=0)
                            y_min = np.min(y_unc,axis=0)
                            # if np.all(np.isfinite(y_max)) & np.all(np.isfinite(y_min)): 
                            #     axs_data[i_plot].fill_between(T_smooth, y_min, y_max, color=colors_fits[2], alpha=.2)
                            
                        except: print('could not solve DPL for {}'.format(feat))    
                        if i_solve == len(p0s)-1: solving=False # you didn't solve it, but it's time to move on
                    
                # housekeeping to make the plots look nice
                axs_data[i_plot].set_ylabel(name_plot[i_plot])
                
                # DPL makes axis zoom out too much. specify zoom. 
                y_min = min(data-uc_data.to_numpy(float))
                if data_labfit[i_plot][:2] == 'sd': 
                    if y_min < 0: y_min = -0.01
                
                y_max = max(data+uc_data.to_numpy(float))
                axs_data[i_plot].set_ylim(y_min-0.15*np.abs(y_min), y_max+0.15*np.abs(y_max))
                
                axs_data[i_plot].legend(framealpha=1, edgecolor='black', fontsize=10)
                
                axs_data[i_plot].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
                axs_data[i_plot].xaxis.set_minor_locator(AutoMinorLocator(5))
                axs_data[i_plot].yaxis.set_minor_locator(AutoMinorLocator(5))
            
        
        axs_data[2].set_xticks(np.arange(300, 1301, 200))
        axs_data[2].set_xlim(200, 1400)
        axs_data[2].set_xlabel('Temperature (K)')
        
                
        axs_data[0].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_data[0].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_data[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.setp(axs_data[0].get_xticklabels(), visible=False) 

        axs_data[1].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_data[1].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_data[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.setp(axs_data[1].get_xticklabels(), visible=False)
        
        axs_data[2].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_data[2].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_data[2].yaxis.set_minor_locator(AutoMinorLocator(5))
        

        # plot transimission data
        linewidth = 2
        colors_trans = ['k', '#0028ff','#0080af','#117d11','#be961e','#ff0000',     '#e6ab02', '#fee9ac']
        nu_center = df_feat.nu
        
        # identify neighbors (for air transitions)
        df_neighbors = df_calcs[(df_calcs.nu > nu_center-nu_spacing)&
                                (df_calcs.nu < nu_center+nu_spacing)&
                                (df_calcs.ratio_max > max(df_feat[df_feat.index.str.startswith('ratio_')])-1)]
        
        # axs_trans[0].plot(df_neighbors.nu, df_neighbors.ratio_max/100 + .95,'x',color='m',markersize=10)
        # axs_trans[0].plot(df_neighbors.nu, df_neighbors.ratio_max/100 + .95,'+',color='m',markersize=20)
        
        if feat in features_plot:
            nu_min = nu_center - nu_span[features_plot.index(feat)]
            nu_max = nu_center + 2*nu_span[features_plot.index(feat)]
        
        else: 
            nu_min = nu_center - nu_spacing
            nu_max = nu_center + 2*nu_spacing

        
        y_min = 1
        
        for i_trans, which_file in enumerate(which_files): 
            
            axs_trans[0].plot(wvn[i_trans],trans[i_trans], color=colors_trans[i_trans], label=which_file, 
                      linewidth=linewidth)
            axs_trans[0].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10)

            [istart, istop] = td.bandwidth_select_td(wvn[i_trans], [nu_min, nu_max], max_prime_factor=500, print_value=False)
            
            try: 
                if y_min == 1: y_min = min(trans[i_trans][istart:istop])
                else: y_min = min(y_min, min(trans[i_trans][istart:istop]))
            except: pass

            
            axs_trans[1].plot(wvn[i_trans],res_HT[i_trans], color=colors_trans[i_trans], label=which_file, 
                      linewidth=linewidth)
            
            axs_trans[2].plot(wvn[i_trans],res_updated[i_trans], color=colors_trans[i_trans], label=which_file, 
                      linewidth=linewidth)
            

            
            
        # set axis
        axs_trans[0].set_xlim(nu_min, nu_max)

        if feat in features_plot:
            if d_type == 'pure': axs_trans[0].set_ylim(y_span0[features_plot.index(feat)], 1.02)
            else: axs_trans[0].set_ylim(y_span0[features_plot.index(feat)], 1.002)
            
            axs_trans[1].set_ylim(y_span1[features_plot.index(feat)])
            axs_trans[2].set_ylim(y_span1[features_plot.index(feat)])
        else: 
            axs_trans[0].set_ylim(y_min-(1-y_min)/50, 1.02)
            axs_trans[1].set_ylim(-0.08, 0.08)
            axs_trans[2].set_ylim(-0.08, 0.08)
        
        
        #  remove x label on upper plots (mostly covered)
        plt.setp(axs_trans[0].get_xticklabels(), visible=False) 
        plt.setp(axs_trans[1].get_xticklabels(), visible=False) 
        
        

        # labels
        axs_trans[2].set_xlabel('Wavenumber - {}'.format(int(nu_center)) + ' ($\mathregular{cm^{-1}}$)')
        
        axs_trans[0].set_ylabel('Measured\nTransmission')
        axs_trans[1].set_ylabel('Meas-\nHITRAN')
        axs_trans[2].set_ylabel('Meas-\nMSF TW')

        axs_trans[0].yaxis.set_label_position("right")
        axs_trans[1].yaxis.set_label_position("right")
        axs_trans[2].yaxis.set_label_position("right")
        
        axs_trans[0].yaxis.tick_right()
        axs_trans[1].yaxis.tick_right()
        axs_trans[2].yaxis.tick_right()
        
        
        # add ticks and minor ticks all over
        axs_trans[0].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_trans[0].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_trans[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        axs_trans[0].xaxis.get_offset_text().set_visible(False)
        
        axs_trans[1].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_trans[1].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_trans[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        axs_trans[1].xaxis.get_offset_text().set_visible(False)
        
        axs_trans[2].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_trans[2].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_trans[2].yaxis.set_minor_locator(AutoMinorLocator(5))
        axs_trans[2].xaxis.get_offset_text().set_visible(False)
        
        

        
        if (feat == 12952) and (d_type == 'pure'): 
            axs_data[0].legend(loc = 'upper right', framealpha=1, edgecolor='black', fontsize=10)   
            axs_data[1].legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)        
            axs_data[2].legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)  
            
            axs_data[0].set_ylim((0.08, 0.55))
            axs_data[1].set_ylim((-0.064, 0.025))
            axs_data[2].set_ylim((-0.21, 0.29))
                
        if feat == 14817: 
            axs_data[0].legend(loc = 'upper right', framealpha=1, edgecolor='black', fontsize=10)   
            axs_data[1].legend(loc = 'upper right', framealpha=1, edgecolor='black', fontsize=10)        
            axs_data[2].legend(loc = 'upper right', framealpha=1, edgecolor='black', fontsize=10)    

        if (feat == 13950) and (d_type == 'air'):
            
            axs_data[1].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10)        
            
            axs_data[1].set_ylim((-0.033, 0.0015))
            axs_data[2].set_ylim((0.14, 0.63))
        
        if (feat == 17112) and (d_type == 'air'):
            
            axs_data[0].legend(loc = 'upper right', framealpha=1, edgecolor='black', fontsize=10, ncol=2)
            axs_data[1].legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)        
            axs_data[2].legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)        
            
            axs_data[0].set_ylim((-0.019, 0.15))
            axs_data[1].set_ylim((-0.019, 0.001))
            axs_data[2].set_ylim((0.04, 0.25))  
            
        if (feat == 27034) and (d_type == 'air'):
            
            axs_data[1].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10)        
            
            axs_data[1].set_ylim((-0.033, 0.0015))
            axs_data[2].set_ylim((0.14, 0.63))
            
            

        
        axs_data[0].text(0.015, 0.88, "A", fontsize=12, transform=axs_data[0].transAxes) 
        
        if (feat in [17112, 13950]) and (d_type == 'air'):
            axs_data[1].text(0.015, 0.85, "B", fontsize=12, transform=axs_data[1].transAxes) 
        else: 
            axs_data[1].text(0.015, 0.88, "B", fontsize=12, transform=axs_data[1].transAxes) 
            
        axs_data[2].text(0.015, 0.88, "C", fontsize=12, transform=axs_data[2].transAxes) 
        
        if ((feat == 12952) and (d_type == 'pure')) or ((feat in [17112]) and (d_type == 'air')): 
            axs_trans[0].text(0.015, 0.87, "D", fontsize=12, transform=axs_trans[0].transAxes) 
        elif ((feat in [13950]) and (d_type == 'air')): 
            axs_trans[0].text(0.015, 0.89, "D", fontsize=12, transform=axs_trans[0].transAxes) 
        else: 
            axs_trans[0].text(0.015, 0.95, "D", fontsize=12, transform=axs_trans[0].transAxes) 
        
        if (feat in [17112]) and (d_type == 'air'): 
            axs_trans[1].text(0.015, 0.5, "E", fontsize=12, transform=axs_trans[1].transAxes) 
            axs_trans[2].text(0.015, 0.5, "F", fontsize=12, transform=axs_trans[2].transAxes)
        else: 
            axs_trans[1].text(0.015, 0.75, "E", fontsize=12, transform=axs_trans[1].transAxes) 
            axs_trans[2].text(0.015, 0.75, "F", fontsize=12, transform=axs_trans[2].transAxes) 
        
        
        
        # if partials is False: 
        #     if which: 
        #         plt.savefig(os.path.abspath('')+r'\plots\DPL\with trans\{} {}.png'.format(d_type, feat), bbox_inches='tight',pad_inches = 0.1)
        # else: 
        #     plt.savefig(os.path.abspath('')+r'\plots\DPL\with trans\{} {}.png'.format(d_type, feat), bbox_inches='tight',pad_inches = 0.1)
            
        plt.savefig(os.path.abspath('')+r'\plots\{} {}.svg'.format(d_type, feat), bbox_inches='tight',pad_inches = 0.1)
        
        
        plt.close()
                  
        
        
print(iter_g) 
print(iter_d)
print(iter_sd)
        

#%% plot transmission only (Part III)

d_type = 'pure'

plot_4 = False

plt.close()

if d_type == 'pure': 

    features_plot = [17300, 18406, 19055, 32958, 33706, 34617, 35005, 35251]    
    nu_span =  0.1
    y_min_plot = -0.02
    y_max_plot = 1.02
    y_span1 = [-0.075, 0.075]

    
    which_files = which_files_pure.copy()
    wvn = wvn_pure.copy()
    trans = trans_pure.copy()
    res_updated = res_updated_pure.copy()
    res_HT = res_HT_pure.copy()

elif d_type == 'air':
    
    features_plot = [17300, 18406, 19055, 32958, 33706, 34617, 35005, 35251]    
    nu_span =  0.165
    y_min_plot = 0.55
    y_max_plot = 1.005
    y_span1 = [-0.075, 0.075]

    
    which_files = which_files_air.copy()
    wvn = wvn_air.copy()
    trans = trans_air.copy()
    res_updated = res_updated_air.copy()
    res_HT = res_HT_air.copy()



width_ratios = np.ones(len(features_plot),int).tolist()
# width_ratios[-1] = 1.5

fig = plt.figure(figsize=(6.5*2, 3.6*2)) 
if plot_4: gs = GridSpec(4, len(features_plot), figure=fig, height_ratios=[4,1,1,1], width_ratios=width_ratios, hspace = 0, wspace = 0) 
else: gs = GridSpec(3, len(features_plot), figure=fig, height_ratios=[4,1,1], width_ratios=width_ratios, hspace = 0, wspace = 0) 

axs_trans = []

for i_feat, feat in  enumerate(features_plot): 

    axs_trans.append([])
        
    axs_trans[i_feat].append(fig.add_subplot(gs[0,i_feat]))
    axs_trans[i_feat].append(fig.add_subplot(gs[1,i_feat], sharex=axs_trans[i_feat][0]))
    axs_trans[i_feat].append(fig.add_subplot(gs[2,i_feat], sharex=axs_trans[i_feat][0]))
    if plot_4: axs_trans[i_feat].append(fig.add_subplot(gs[3,i_feat], sharex=axs_trans[i_feat][0]))
    
    
    feat = int(feat)
    df_feat = df_sceg.loc[feat]   
    
    quanta = df_feat.quanta.split()
    try: kcJ = float(quanta[11])/float(quanta[9])
    except: kcJ = 0.
    
    KaKcp = quanta[7] + ',' + quanta[8]
    KaKcpp = quanta[10] + ',' + quanta[11]
    
    title = '({}$_{{{}}}$←{}$_{{{}}}$)'.format(quanta[6],KaKcp,quanta[9],KaKcpp)
    
    axs_trans[i_feat][0].text(0.05, 0.03, title, fontsize=12, transform=axs_trans[i_feat][0].transAxes) 
    
    if i_feat in [1,7]: 
        axs_trans[i_feat][0].text(0.42, 0.95, ['A','B','C','D','E','F','G','H'][i_feat], fontsize=12, transform=axs_trans[i_feat][0].transAxes) 
    else: 
        axs_trans[i_feat][0].text(0.45, 0.95, ['A','B','C','D','E','F','G','H'][i_feat], fontsize=12, transform=axs_trans[i_feat][0].transAxes) 
    
    # plot transimission data
    linewidth = 2
    colors_trans = ['k', '#0028ff','#0080af','#117d11','#be961e','#ff0000',     '#e6ab02', '#fee9ac']
    
    nu_center = df_feat.nu    
    nu_center_int = int(nu_center)
    
    if d_type == 'air': 
        if i_feat in [0,2,5]: nu_center -= 0.01
        if i_feat == 1: nu_center += 0.01

    
    if i_feat == 0: nu_center_int += 1
    
    nu_min = nu_center - nu_span
    nu_max = nu_center + nu_span

    y_min = 1
    
    for i_trans, which_file in enumerate(which_files): 
        
        axs_trans[i_feat][0].plot(wvn[i_trans]-nu_center_int,trans[i_trans], color=colors_trans[i_trans], label=which_file+'orr', 
                  linewidth=linewidth)
        if (i_feat == len(features_plot)-1) and (d_type == 'pure'): 
            axs_trans[i_feat][0].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10, bbox_to_anchor=(0.8, 0.08))
        if (i_feat == 1) and (d_type == 'air'): 
            axs_trans[i_feat][0].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10, bbox_to_anchor=(0.3, 0.08))

        
        axs_trans[i_feat][1].plot(wvn[i_trans]-nu_center_int,res_HT[i_trans], color=colors_trans[i_trans], label=which_file, 
                  linewidth=linewidth)
        
        axs_trans[i_feat][2].plot(wvn[i_trans]-nu_center_int,res_updated[i_trans], color=colors_trans[i_trans], label=which_file, 
                  linewidth=linewidth)
        
        
    # set axis
    axs_trans[i_feat][0].set_xticks(np.arange(-1,1.4,0.1))
    axs_trans[i_feat][0].set_xlim(nu_min-nu_center_int, nu_max-nu_center_int)

    axs_trans[i_feat][0].set_ylim(y_min_plot, y_max_plot)
    axs_trans[i_feat][1].set_ylim(y_span1)
    axs_trans[i_feat][2].set_ylim(y_span1)
    if plot_4: axs_trans[i_feat][3].set_ylim(y_span1)
    
    
    #  remove x label on upper plots (mostly covered)
    plt.setp(axs_trans[i_feat][0].get_xticklabels(), visible=False) 
    plt.setp(axs_trans[i_feat][1].get_xticklabels(), visible=False) 
    if plot_4: plt.setp(axs_trans[i_feat][2].get_xticklabels(), visible=False) 
    
    #  remove y label on later plots (mostly covered)
    if i_feat > 0: 
        plt.setp(axs_trans[i_feat][0].get_yticklabels(), visible=False) 
        plt.setp(axs_trans[i_feat][1].get_yticklabels(), visible=False) 
        plt.setp(axs_trans[i_feat][2].get_yticklabels(), visible=False) 
        if plot_4: plt.setp(axs_trans[i_feat][3].get_yticklabels(), visible=False)
    
    

    # labels
    axs_trans[i_feat][-1].set_xlabel('Wavenumber\n- {}'.format(nu_center_int) + ' [cm$^{-1}$]')
    
    if i_feat == 0: 
        axs_trans[i_feat][0].set_ylabel('Measured\nTransmission')
        axs_trans[i_feat][1].set_ylabel('Meas-\nHITRAN')
        axs_trans[i_feat][2].set_ylabel('Meas-\nMSF (SPL)')
        if plot_4: axs_trans[i_feat][3].set_ylabel('Meas-\nSTF TW')
   
    
    # add ticks and minor ticks all over
    for i_panel, _ in enumerate(axs_trans[i_feat]): 
    
        axs_trans[i_feat][i_panel].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
        axs_trans[i_feat][i_panel].xaxis.set_minor_locator(AutoMinorLocator(10))
        axs_trans[i_feat][i_panel].yaxis.set_minor_locator(AutoMinorLocator(5))
        axs_trans[i_feat][i_panel].xaxis.get_offset_text().set_visible(False)



plt.savefig(os.path.abspath('')+r'\plots\DPL trans {}.svg'.format(d_type), bbox_inches='tight',pad_inches = 0.1)

# plt.close()
      


#%% DPL plots (Part III) measured values

plt.close()
save_fig = False

d_type = 'pure' # 'pure' or 'air'
prop_plot = 'delta' # gamma delta sd
features_plot = [17300, 18406, 19055, 32958, 33706, 34617, 35005, 35251] # from 45 that met criteria for both air and pure

if d_type == 'pure': 
    if prop_plot == 'gamma': y_span = 0.4
    if prop_plot == 'delta': y_span = 0.08
    if prop_plot == 'sd': y_span = 0.3

elif d_type == 'air': 
    if prop_plot == 'gamma': y_span = 0.08
    if prop_plot == 'delta': y_span = 0.03
    if prop_plot == 'sd': y_span = 0.3


colors_fits = ['lime','darkorange','blue', 'red']



fig = plt.figure(figsize=(6.5, 14)) 
gs = GridSpec(len(features_plot), 1, figure=fig, hspace = 0.02) 
axs_data = []

RMS_labfit = np.zeros_like(features_plot,float) # calculating weighted RMS
RMS_HITRAN = np.zeros_like(features_plot,float)
RMS_DPL = np.zeros_like(features_plot,float)

prop_extractor = np.zeros_like(features_plot,float)

for i_feat, feat in enumerate(features_plot): 
    
    if i_feat == 0: axs_data.append(fig.add_subplot(gs[i_feat]))
    else: axs_data.append(fig.add_subplot(gs[i_feat], sharex=axs_data[0]))
    
    feat = int(feat)
    df_feat = df_sceg.loc[feat]   
    
    prop_extractor[i_feat] = df_calcs.nu[feat]
    
    quanta = df_feat.quanta.split()
    try: kcJ = float(quanta[11])/float(quanta[9])
    except: kcJ = 0.
    
    KaKcp = quanta[7] + ',' + quanta[8]
    KaKcpp = quanta[10] + ',' + quanta[11]
    
    title = '({}$_{{{}}}$←{}$_{{{}}}$)'.format(quanta[6],KaKcp,quanta[9],KaKcpp)
    
    if prop_plot == 'gamma':
        axs_data[i_feat].text(0.02, 0.08, title, fontsize=12, transform=axs_data[i_feat].transAxes) 
        axs_data[i_feat].text(0.95, 0.82, ['A','B','C','D','E','F','G','H'][i_feat], fontsize=12, transform=axs_data[i_feat].transAxes)

    if prop_plot == 'delta':
        axs_data[i_feat].text(0.7, 0.08, title, fontsize=12, transform=axs_data[i_feat].transAxes) 
        axs_data[i_feat].text(0.02, 0.78, ['A','B','C','D','E','F','G','H'][i_feat], fontsize=12, transform=axs_data[i_feat].transAxes)

    if prop_plot == 'sd': 
        axs_data[i_feat].text(0.03, 0.08, title, fontsize=12, transform=axs_data[i_feat].transAxes) 
        axs_data[i_feat].text(0.95, 0.82, ['A','B','C','D','E','F','G','H'][i_feat], fontsize=12, transform=axs_data[i_feat].transAxes)
    
    
    if d_type == 'pure': 
        if prop_plot == 'gamma': 
            name_plot = 'Self-Width, γ$_{self}$\n[cm$^{-1}$/atm]'
            data_HT = ['gamma_self',False]    
            uc_HT = [3,False]
            data_labfit = 'n_self'
            
            df_gamma_self = df_feat[df_feat.index.str.startswith('gamma_self_')]
            df_uc_gamma_self = df_feat[df_feat.index.str.startswith('uc_gamma_self_')]
            df_uc_gamma_self[:] = np.sqrt((df_uc_gamma_self.to_numpy(float)/df_gamma_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_gamma_self
            
            data_plots = [df_gamma_self, df_uc_gamma_self] 
                         
        elif prop_plot == 'delta': 
            name_plot = 'Self-Shift, δ$_{self}$\n[cm$^{-1}$/atm]'
            data_HT = False
            uc_HT = False
            data_labfit = 'n_delta_self'

            df_delta_self = df_feat[df_feat.index.str.startswith('delta_self_')]
            df_uc_delta_self = df_feat[df_feat.index.str.startswith('uc_delta_self_')]
            df_uc_delta_self[:] = np.sqrt((df_uc_delta_self.to_numpy(float)/df_delta_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2 + 
                                          (1.7E-4 / (0.021*df_delta_self.to_numpy(float)))**2) * abs(df_delta_self)
            
            data_plots = [df_delta_self, df_uc_delta_self] 
                
            
        elif prop_plot == 'sd': 
            name_plot = 'SD of the\nSelf-Width, a$_{w}$'
            data_HT = False
            uc_HT = False
            data_labfit = 'sd_self'

            df_sd_self = df_feat[df_feat.index.str.startswith('sd_self_')]
            df_uc_sd_self = df_feat[df_feat.index.str.startswith('uc_sd_self_')]
            df_uc_sd_self[:] = np.sqrt((df_uc_sd_self.to_numpy(float)/df_sd_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_sd_self # droped 3.9% due to TD of SD
        
            data_plots = [df_sd_self, df_uc_sd_self] 


    elif d_type == 'air': 
        if prop_plot == 'gamma': 
            name_plot = 'Air-Width, γ$_{air}$\n[cm$^{-1}$/atm]'
            data_HT = ['gamma_air', 'n_air']  
            uc_HT = [2,4]
            data_labfit = 'n_air'
            
            df_gamma_air = df_feat[df_feat.index.str.startswith('gamma_air_')]
            df_uc_gamma_air = df_feat[df_feat.index.str.startswith('uc_gamma_air_')]
            df_uc_gamma_air[:] = np.sqrt((df_uc_gamma_air.to_numpy(float)/df_gamma_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_gamma_air
              
            data_plots = [df_gamma_air, df_uc_gamma_air]       
              
        elif prop_plot == 'delta': 
            name_plot = 'Air-Shift, δ$_{air}$\n[cm$^{-1}$/atm]'
            data_HT = ['delta_air', False]
            uc_HT = [5, False]
            data_labfit = 'n_delta_air'

            df_delta_air = df_feat[df_feat.index.str.startswith('delta_air_')]
            df_uc_delta_air = df_feat[df_feat.index.str.startswith('uc_delta_air_')]
            df_uc_delta_air[:] = np.sqrt((df_uc_delta_air.to_numpy(float)/df_delta_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 +  (y_unc_air)**2 +
                                          (1.7E-4 / (0.789*df_delta_air.to_numpy(float)))**2) * abs(df_delta_air)

            data_plots = [df_delta_air, df_uc_delta_air]
            
            
        elif prop_plot == 'sd': 
            name_plot = 'SD of the\nAir-Width, a$_{w}$'
            data_HT = False
            uc_HT = False
            data_labfit = 'sd_air'

            df_sd_air = df_feat[df_feat.index.str.startswith('sd_air_')]
            df_uc_sd_air = df_feat[df_feat.index.str.startswith('uc_sd_air_')]
            df_uc_sd_air[:] = np.sqrt((df_uc_sd_air.to_numpy(float)/df_sd_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_sd_air # droped 5.2
            
            data_plots = [df_sd_air, df_uc_sd_air]

    # plot the data for current transition
    
    [data, uc_data]  = data_plots
            
    axs_data[i_feat].plot(T_conditions, data, color='k', marker='x', markersize=10, markeredgewidth=3, linestyle='None', label='Measurement', zorder=10)
    axs_data[i_feat].errorbar(T_conditions, data, uc_data, color='k', fmt='none', capsize=5, zorder=10)

        
    # overlay HITRAN prediction
    if data_HT: 
        
        base = df_feat[data_HT[0] + 'HT']
        if data_HT[1]:
            n = df_feat[data_HT[1] + 'HT']
        else: 
            n=0
        
        if uc_HT: 
            uc_base_str = df_feat.ierr[uc_HT[0]]
            
            if HT_uncertainty[uc_base_str]: 
                uc_base = base * HT_uncertainty[uc_base_str]
            else: uc_base = 0
            
            if uc_HT[1]: 
                uc_n_str = df_feat.ierr[uc_HT[1]]
                                
                if HT_uncertainty[uc_n_str]: 
                    uc_n = n * HT_uncertainty[uc_n_str]
                else: uc_n = 0
            else: uc_n = 0
                
        # print('base = {} n = {}'.format(uc_base_str, uc_n_str))
            
        y_center = SPL(T_smooth, base, n)
        y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                        SPL(T_smooth, base+uc_base, n-uc_n), 
                        SPL(T_smooth, base-uc_base, n+uc_n),
                        SPL(T_smooth, base-uc_base, n-uc_n)])
        y_max = np.max(y_unc,axis=0)
        y_min = np.min(y_unc,axis=0)
        
        axs_data[i_feat].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN')
        axs_data[i_feat].fill_between(T_smooth, y_min, y_max, color=colors_fits[3], alpha=.2)
    
        RMS_HITRAN[i_feat] = np.sqrt(np.mean(((SPL(T_conditions, base, n) - data)/uc_data.to_list())**2))
        
        HT = SPL(T_conditions, base, n) 
        print()
        print(HT[0])
        # print(data[0])
        # print(uc_data.to_list()[0])
    
    # overlay Labfit prediction (if there is one)
    if df_feat['uc_'+data.index[0][:-4]] != -1: 
        
        color_labfit = colors_fits[0]
        linestyle = 'solid'
        
        if data_labfit[:2] == 'sd':  
           
            base = df_feat[data_labfit]
            n = 0
            
            y_center = base * np.ones_like(T_smooth)
            y_max = y_center + df_feat['uc_'+data_labfit]
            y_min = y_center - df_feat['uc_'+data_labfit]
            
        else:     
           
            base = df_feat[data.index[0][:-4]]
            uc_base = df_feat['uc_'+data.index[0][:-4]]
            
            n = df_feat[data_labfit]
            uc_n = df_feat['uc_'+data_labfit]
            if uc_n == -1: 
                uc_n = 0  
                color_labfit = 'darkgreen'
                linestyle = 'dashed'

            # if prop_plot == 'delta': delta_nu = df_feat.nu - df_feat.nu_300                             
            
            y_center = SPL(T_smooth, base, n) 
            y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                            SPL(T_smooth, base+uc_base, n-uc_n), 
                            SPL(T_smooth, base-uc_base, n+uc_n),
                            SPL(T_smooth, base-uc_base, n-uc_n)])
            y_max = np.max(y_unc,axis=0)
            y_min = np.min(y_unc,axis=0)
                                
        axs_data[i_feat].plot(T_smooth, y_center, color=color_labfit, label='MSF (SPL)', linestyle=linestyle)
        axs_data[i_feat].fill_between(T_smooth, y_min, y_max, color=color_labfit, alpha=.2)
    
        RMS_labfit[i_feat] = np.sqrt(np.mean(((SPL(T_conditions, base, n) - data)/uc_data.to_list())**2))
        
    if data_labfit.split('_')[1] == 'delta': 
        axs_data[i_feat].plot(T_smooth,np.zeros_like(T_smooth), color='k', linestyle=':')
    
    if (df_feat['uc_'+data.index[0][:-4]] == -1)&(data_labfit[:2] == 'sd'):
        y_center = df_feat[data_labfit] * np.ones_like(T_smooth)
        axs_data[i_feat].plot(T_smooth, y_center, color='darkgreen', label='MSF (SPL)', linestyle='dashed')
                
    if data_labfit[:2] != 'sd': 

        # fit the data using DPL (weighted by Labfit uncertainties)
        solving=True
        i_solve=-1
        
        if df_feat['uc_' + data_labfit] == -1: n = 0.75
        
        p0s = [[2*data[0], n, -2*data[0], -1*n], [data[0], n, -1*data[0], -1*n], [data[0]/2, n, data[0]/2, -1*n], [data[0], n, 0, 0]]
        while solving:
            i_solve+=1
            try: 
                p_DPL, cov = fit(DPL, T_conditions, data, p0=p0s[i_solve], maxfev=15000, sigma=uc_data.to_numpy(float))
                p_err = np.sqrt(np.diag(cov))

                solving=False # you solved it!
                
                if p_DPL[2] > 0: sign = '+'
                else: sign = ''
                axs_data[i_feat].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], label='DPL')
                
                # calculate uncertainties (too many Infs and NANS - haven't been using much)
                y_unc = np.array([DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                  DPL(T_smooth, p_DPL[0]+p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]+p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]+p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]+p_err[2], p_DPL[3]-p_err[3]), 
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]+p_err[3]),
                                  DPL(T_smooth, p_DPL[0]-p_err[0], p_DPL[1]-p_err[1], p_DPL[2]-p_err[2], p_DPL[3]-p_err[3])])
                y_max = np.max(y_unc,axis=0)
                y_min = np.min(y_unc,axis=0)
                # if np.all(np.isfinite(y_max)) & np.all(np.isfinite(y_min)): 
                #     axs_data[i_feat].fill_between(T_smooth, y_min, y_max, color=colors_fits[2], alpha=.2)
                
            except: print('could not solve DPL for {}'.format(feat))    
            if i_solve == len(p0s)-1: solving=False # you didn't solve it, but it's time to move on
        
        RMS_DPL[i_feat] = np.sqrt(np.mean(((DPL(T_conditions, *p_DPL) - data)/uc_data.to_list())**2))
        
    
    # housekeeping to make the plots look nice
    axs_data[i_feat].set_ylabel(name_plot)

    if prop_plot == 'sd': axs_data[i_feat].set_yticks(np.arange(0.0, 0.6, 0.1)) 
    
    # specify zoom. 
    y_mid = np.mean([min(data-uc_data.to_numpy(float)), max(data+uc_data.to_numpy(float))])
    
    if (d_type=='air')&(prop_plot=='sd')&(i_feat==7): y_mid -=0.02
    
    y_min = y_mid - y_span/2
    y_max = y_mid + y_span/2

    axs_data[i_feat].set_ylim(y_min, y_max)
    

    if i_feat == 0: 
        if prop_plot == 'sd':        
            axs_data[i_feat].legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=10, ncol=4, bbox_to_anchor=(0.793, 1.27))
        elif (prop_plot == 'delta')&(d_type == 'pure'):
            axs_data[i_feat].legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=10, ncol=4, bbox_to_anchor=(0.89, 1.27))
        else: 
            axs_data[i_feat].legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=10, ncol=4, bbox_to_anchor=(1.0, 1.27))
    
    axs_data[i_feat].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
    axs_data[i_feat].xaxis.set_minor_locator(AutoMinorLocator(10))
    axs_data[i_feat].yaxis.set_minor_locator(AutoMinorLocator(5))
    if i_feat != len(features_plot)-1: plt.setp(axs_data[i_feat].get_xticklabels(), visible=False) 


    
axs_data[-1].set_xticks(np.arange(300, 1301, 200))
axs_data[-1].set_xlim(200, 1400)
axs_data[-1].set_xlabel('Temperature (K)')
        



if save_fig: plt.savefig(os.path.abspath('')+r'\plots\DPL {} {}.svg'.format(d_type, prop_plot), bbox_inches='tight',pad_inches = 0.1)

# plt.close()
          



