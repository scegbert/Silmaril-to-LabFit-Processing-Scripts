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

d_type = 'air' # 'pure' or 'air'

d_sceg_save = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'


#%% load in the DPL data


# for param in extra_params: 
#     df_sceg[param] = df_sceg2[param]


f = open(os.path.join(d_sceg_save,'DPL results.pckl'), 'rb')
[df_sceg, T_conditions, features_strong, features_doublets] = pickle.load(f)
f.close()     

T_conditions = [float(T) for T in T_conditions]
T_conditions = np.asarray(T_conditions)



#%% set up fit equations

def SPL(T, c, n): 
    
    return c*(296/T)**n

def SPLoff(T, c1, c2, n): 

    return c1*(296/T)**n + c2
    
def DPL(T, c1, n1, c2, n2): 
    
    return c1*(296/T)**n1 + c2*(296/T)**n2


#%% load in transmission data (model from labfit results)

# load in labfit stuff (transmission, wvn, residuals before and after, conditions)
d_sceg = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_HT_pure, res_sd0] = pickle.load(f)
f.close()

[T_all, P_all] = np.asarray([T_pure, P_pure])


#%% plot and verify this is what you want

wvn2_data = [6600, 7600]

which_files = ['300 K 16 T', '500 K 16 T', '700 K 16 T', '900 K 16 T', '1100 K 16 T', '1300 K 16 T']

res_HT = []
res_og = []
res_updated = []
trans = []
wvn = []

for which_file in which_files: 

    T_plot = int(which_file.split()[0])
    P_plot = int(which_file.split()[2])
    
    i_plot = np.where((T_all == T_plot) & (P_all == P_plot))[0]
        
    T = [T_all[i] for i in i_plot]
    P = [P_all[i] for i in i_plot]
    
    wvn_labfit_all = np.concatenate([wvn_pure[i] for i in i_plot])
    trans_labfit_all = np.concatenate([trans_pure[i] for i in i_plot])
    res_updated_all = np.concatenate([res_pure[i] for i in i_plot])
    res_HT_all = np.concatenate([res_HT_pure[i] for i in i_plot])
    
    
    [istart, istop] = td.bandwidth_select_td(wvn_labfit_all, wvn2_data, max_prime_factor=500, print_value=False)
    
    wvn.append(wvn_labfit_all[istart:istop])
    trans.append(trans_labfit_all[istart:istop]/100)
    res_updated.append(res_updated_all[istart:istop]/100)
    res_HT.append(res_HT_all[istart:istop]/100) 

    # plt.plot(wvn[-1], trans[-1], label='updated')
    # plt.plot(wvn[-1], res_updated[-1], label='og')
    # plt.plot(wvn[-1], res_HT[-1], label='HT')

# plt.legend()

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


please = stophere_fullyloaded

#%% DPL plots with transmission (only for air or pure, not both at the same time)


if d_type == 'pure': 

    features_plot = [14817, 12952]
    nu_span =  [0.06, 0.08]
    y_span0 = [0.76, 0.21]
    y_span1 = [[-0.049, 0.02], [-0.075, 0.031]]

    # features_plot = [14817, 7094, 35036, 12952, 26463]
    # nu_span =  [0.06, 0.06, 0.08, 0.08]
    # y_span0 = [0.76, 0.67, 0.12, 0.21]
    # y_span1 = [[-0.049, 0.02], [-0.075, 0.036], [-0.075, 0.041], [-0.075, 0.031]]

elif d_type == 'air':
    
    features_plot = [17112, 13950]
    nu_span =  [0.08, 0.08]
    y_span0 = [0.44, 0.701]
    y_span1 = [[-0.055, 0.019], [-0.04, 0.017]]


T_smooth = np.linspace(T_conditions[0]-110, T_conditions[-1]+110, num=500)
df_quanta = db.labfit_to_df('E:\water database\pure water\B1\B1-000-HITRAN')

T_unc_pure = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 19/1288])
T_unc_air = np.array([2/295, 4/505, 5/704, 8/901, 11/1099, 20/1288])

P_unc_pure = 0.0027
P_unc_air = 0.0025
y_unc_air = 0.029

colors_fits = ['lime','darkorange','blue', 'red']



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


for i_feat, feat in  enumerate(features_plot): #enumerate(df_sceg.index): # 
    
    feat = int(feat)
    
    df_feat = df_sceg.loc[feat]   
    
    df_feat = df_sceg.loc[feat]   
       

    if d_type == 'pure': 
        pure_T_index = [index for index in df_feat.index if '_self_' in index]
        which = ~np.any(df_feat[pure_T_index]==-1)
    elif d_type == 'air': 
        air_T_index = [index for index in df_feat.index if '_air_' in index]
        which = ~np.any(df_feat[air_T_index]==-1)
        
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
            data_plots = [[df_gamma_self, df_uc_gamma_self], 
                          [df_delta_self, df_uc_delta_self], 
                          [df_sd_self, df_uc_sd_self]]
            
        elif d_type == 'air': 
            data_plots = [[df_gamma_air, df_uc_gamma_air], 
                          [df_delta_air, df_uc_delta_air], 
                          [df_sd_air, df_uc_sd_air]]

        
        for i_plot, [data, uc_data] in enumerate(data_plots): 
                    
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
                
                if data_HT[i_plot][0] == 'gamma_self': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}'.format(base))
                elif data_HT[i_plot][0] == 'gamma_air': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                elif data_HT[i_plot][0] == 'delta_air': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f} (no TD)'.format(base))
                axs_data[i_plot].fill_between(T_smooth, y_min, y_max, color=colors_fits[3], alpha=.2)
                    
            # overlay Labfit prediction (if there is one)
            if df_feat['uc_'+data.index[0][:-4]] != -1: 
                
                color_labfit = colors_fits[0]
                
                if data_labfit[i_plot][:2] == 'sd':  
                   
                    sd = df_feat[data_labfit[i_plot]]
                    
                    y_center = sd * np.ones_like(T_smooth)
                    y_max = y_center + df_feat['uc_'+data_labfit[i_plot]]
                    y_min = y_center - df_feat['uc_'+data_labfit[i_plot]]
                    
                else:                       
                    
                    base = df_feat[data.index[0][:-4]]
                    uc_base = df_feat['uc_'+data.index[0][:-4]]
                    
                    n = df_feat[data_labfit[i_plot]]
                    uc_n = df_feat['uc_'+data_labfit[i_plot]]
                    if uc_n == -1: 
                        uc_n = 0  
                        color_labfit = 'darkgreen'
                    
                    y_center = SPL(T_smooth, base, n)
                    y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                    SPL(T_smooth, base+uc_base, n-uc_n), 
                                    SPL(T_smooth, base-uc_base, n+uc_n),
                                    SPL(T_smooth, base-uc_base, n-uc_n)])
                    y_max = np.max(y_unc,axis=0)
                    y_min = np.min(y_unc,axis=0)
                                        
                if data_labfit[i_plot][:2] == 'sd': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=color_labfit, label='Multispectrum Fit TW a$_w$ = {:.2f}'.format(sd))
                else: 
                    axs_data[i_plot].plot(T_smooth, y_center, color=color_labfit, label='Multispectrum Fit TW {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
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
                        
                        if p_DPL[2] > 0: sign = '+'
                        else: sign = ''
                        axs_data[i_plot].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], 
                                              label='DPL Fit TW {:.3f}(T/T)$^{{{:.2f}}}${}{:.3f}(T/T)$^{{{:.2f}}}$'.format(p_DPL[0],p_DPL[1],sign,p_DPL[2],p_DPL[3]))
                        
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
        
        if feat in features_plot:
            nu_min = nu_center - nu_span[features_plot.index(feat)]
            nu_max = nu_center + 2*nu_span[features_plot.index(feat)]
        
        else: 
            nu_min = nu_center - 0.08 
            nu_max = nu_center + 2*0.08 

        
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
            axs_trans[0].set_ylim(y_span0[features_plot.index(feat)], 1.02)
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

        
        axs_data[0].text(0.015, 0.88, "A", fontsize=12, transform=axs_data[0].transAxes) 
        
        if (feat in [17112, 13950]) and (d_type == 'air'):
            axs_data[1].text(0.015, 0.88, "B", fontsize=12, transform=axs_data[1].transAxes) 
        else: 
            axs_data[1].text(0.015, 0.88, "B", fontsize=12, transform=axs_data[1].transAxes) 
            
        axs_data[2].text(0.015, 0.88, "C", fontsize=12, transform=axs_data[2].transAxes) 
        
        if ((feat == 12952) and (d_type == 'pure')) or ((feat in [17112]) and (d_type == 'air')): 
            axs_trans[0].text(0.015, 0.87, "D", fontsize=12, transform=axs_trans[0].transAxes) 
        else: 
            axs_trans[0].text(0.015, 0.95, "D", fontsize=12, transform=axs_trans[0].transAxes) 
                
        axs_trans[1].text(0.015, 0.5, "E", fontsize=12, transform=axs_trans[1].transAxes) 
        axs_trans[2].text(0.015, 0.8, "F", fontsize=12, transform=axs_trans[2].transAxes) 
        
        
        


        # plt.savefig(os.path.abspath('')+r'\plots\DPL\with trans\{} {}.png'.format(d_type, feat), bbox_inches='tight',pad_inches = 0.1)
        # plt.savefig(os.path.abspath('')+r'\plots\{} {}.svg'.format(d_type, feat), bbox_inches='tight',pad_inches = 0.1)
        
        # plt.close()
                  






























