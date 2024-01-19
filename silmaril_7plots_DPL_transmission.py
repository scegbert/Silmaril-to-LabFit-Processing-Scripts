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

d_type = 'pure' # 'pure' or 'air'

d_sceg_save = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'


#%% load in the DPL data


# for param in extra_params: 
#     df_sceg[param] = df_sceg2[param]


f = open(os.path.join(d_sceg_save,'DPL results.pckl'), 'rb')
[df_sceg, T_conditions, features_strong, features_doublets] = pickle.load(f)
f.close()     

types_RMS = ['labfit', 'HT', 'SPL', 'DPL']  
d_whiches = ['self', 'air']
extra_params_base = ['gamma','sd','delta'] # self and air width, SD, and shift

# RMS for all values
columns_RMS = []
for d_which in d_whiches: 
    for param in extra_params_base: 
        for type_RMS in types_RMS:
            
            name = 'RMS_' + type_RMS + '_' + param + '_' + d_which
            columns_RMS.append(name)

# check for over conditioning for only scipy calculated values
for d_which in d_whiches: 
    for param in extra_params_base: 
        for type_RMS in types_RMS[2:]: 
            
            name = 'condition_' + type_RMS + '_' + param + '_' + d_which
            columns_RMS.append(name)

df_sceg = pd.merge(df_sceg, df_sceg.reindex(columns=columns_RMS), left_index=True, right_index=True)

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



please = stophere_fullyloaded

#%% DPL plots with transmission (only for air or pure, not both at the same time)


features_plot = [14817, 15596]
nu_span =  [[6920.468, 6920.67]] # [[6920.468, 6920.67]]
y_span0 = [[]]

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
    
    data_HT = [['gamma_self','n_air'], 
               False, 
               False]

    data_labfit = ['n_self', 
                   'n_delta_self', 
                   'sd_self']


elif d_type == 'air': 
    name_plot = ['Air-Width\nγ$_{air}$ [cm$^{-1}$/atm]', 
                 'Air-Shift\nδ$_{air}$ [cm$^{-1}$/atm]', 
                 'SD Air-Width\nγ$_{SD,air}$  [cm$^{-1}$/atm]']

    data_HT = [['gamma_air', 'n_air'], 
               ['delta_air', False], 
               False]
    
    data_labfit = ['n_air', 
                   'n_delta_air', 
                   'sd_air']


for i_feat, feat in enumerate(df_sceg.index): 
    
    feat = int(feat)
    
    df_feat = df_sceg.loc[feat]   
    
    # confirm all values are floated and not doublets (overly restrictive for now, will ease up later)
    if feat in features_plot: # ~np.any(df_feat[-74:]==-1): 

        # df_gamma_self = df_feat[df_feat.index.str.startswith('gamma_self_')]
        # df_uc_gamma_self = df_feat[df_feat.index.str.startswith('uc_gamma_self_')]
        
        
        # df_uc_gamma_self[:] = np.sqrt((df_uc_gamma_self.to_numpy(float)/df_gamma_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_gamma_self
        
        # df_sd_self = df_feat[df_feat.index.str.startswith('sd_self_')]
        # df_uc_sd_self = df_feat[df_feat.index.str.startswith('uc_sd_self_')]
        # df_uc_sd_self[:] = np.sqrt((df_uc_sd_self.to_numpy(float)/df_sd_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2) * df_sd_self # droped 3.9% due to TD of SD
        
        # df_delta_self = df_feat[df_feat.index.str.startswith('delta_self_')]
        # df_uc_delta_self = df_feat[df_feat.index.str.startswith('uc_delta_self_')]
        # df_uc_delta_self[:] = np.sqrt((df_uc_delta_self.to_numpy(float)/df_delta_self.to_numpy(float))**2 + (P_unc_pure)**2 + (T_unc_pure)**2 + 
        #                               (1.7E-4 / (0.021*df_delta_self.to_numpy(float)))**2) * abs(df_delta_self)
                                      
        
        # df_gamma_air = df_feat[df_feat.index.str.startswith('gamma_air_')]
        # df_uc_gamma_air = df_feat[df_feat.index.str.startswith('uc_gamma_air_')]
        # df_uc_gamma_air[:] = np.sqrt((df_uc_gamma_air.to_numpy(float)/df_gamma_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_gamma_air
                
        # df_sd_air = df_feat[df_feat.index.str.startswith('sd_air_')]
        # df_uc_sd_air = df_feat[df_feat.index.str.startswith('uc_sd_air_')]
        # df_uc_sd_air[:] = np.sqrt((df_uc_sd_air.to_numpy(float)/df_sd_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 + (y_unc_air)**2) * df_sd_air # droped 5.2
        
        # df_delta_air = df_feat[df_feat.index.str.startswith('delta_air_')]
        # df_uc_delta_air = df_feat[df_feat.index.str.startswith('uc_delta_air_')]
        # df_uc_delta_air[:] = np.sqrt((df_uc_delta_air.to_numpy(float)/df_delta_air.to_numpy(float))**2 + (P_unc_air)**2 + (T_unc_air)**2 +  (y_unc_air)**2 +
        #                               (1.7E-4 / (0.789*df_delta_air.to_numpy(float)))**2) * abs(df_delta_air)
        
        
        
        fig = plt.figure(figsize=(10, 8)) 
        gs = GridSpec(4, 2, figure=fig, width_ratios=[2,1], height_ratios=[2,2,1,1]) 
        axs_data = []
        axs_data.append(fig.add_subplot(gs[0,0]))
        axs_data.append(fig.add_subplot(gs[1,0]))
        axs_data.append(fig.add_subplot(gs[2:,0]))
        
        axs_trans = []
        axs_trans.append(fig.add_subplot(gs[:2,1]))
        axs_trans.append(fig.add_subplot(gs[2,1]))
        axs_trans.append(fig.add_subplot(gs[3,1]))
                
        fig.subplots_adjust(top=0.95, wspace=0.01)
        plt.show()

        
        quanta = df_feat.quanta.split()
        try: kcJ = float(quanta[8])/float(quanta[6])
        except: kcJ = 0.
               
        title = '{}{}{} ← {}{}{}      {}$_{}$$_{}$ ← {}$_{}$$_{}$      Kc" / J" = {:.3f}'.format(quanta[0],quanta[1],quanta[2],quanta[3],quanta[4],quanta[5],
                                                                        quanta[6],quanta[7],quanta[8],quanta[9],quanta[10],quanta[11], kcJ)
        
        plt.suptitle(title)
        
        
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
                
                y_center = SPL(T_smooth, base, n)
                
                if data_HT[i_plot][0] == 'gamma_self': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$ (using n$_{{{}}}$)'.format(base,n,'air'))
                elif data_HT[i_plot][0] == 'gamma_air': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                elif data_HT[i_plot][0] == 'delta_air': 
                    axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[3], label='HITRAN {:.3f} (no TD)'.format(base))
                    
                df_sceg.loc[feat, 'RMS_HT_'+data.index[0][:-4]] = np.sqrt(np.sum((SPL(T_conditions, base, n) - data)**2))
                    
            
            # overlay Labfit prediction (if there is one)
            if df_feat['uc_' + data_labfit[i_plot]] != -1: 
                
                if data_labfit[i_plot][:2] == 'sd':  
                   
                    base = df_feat[data.index[0][:-4]] * df_feat['gamma'+data_labfit[i_plot][2:]]
                    uc_base = df_feat['uc_'+data.index[0][:-4]]
                    
                    n = df_feat['n'+data_labfit[i_plot][2:]]
                    uc_n = df_feat['uc_n'+data_labfit[i_plot][2:]]
                    
                    # data *= data_plots[0?][0].to_numpy(float)
                    
                else:                       
                    
                    base = df_feat[data.index[0][:-4]]
                    uc_base = df_feat['uc_'+data.index[0][:-4]]
                    
                    n = df_feat[data_labfit[i_plot]]
                    uc_n = df_feat['uc_'+data_labfit[i_plot]]
                
                y_center = SPL(T_smooth, base, n)
                y_unc = np.array([SPL(T_smooth, base+uc_base, n+uc_n), 
                                SPL(T_smooth, base+uc_base, n-uc_n), 
                                SPL(T_smooth, base-uc_base, n+uc_n),
                                SPL(T_smooth, base-uc_base, n-uc_n)])
                y_max = np.max(y_unc,axis=0)
                y_min = np.min(y_unc,axis=0)
                
                if data_labfit[i_plot][:2] == 'sd': axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[0], label='Labfit (from a$_w$), {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                else: axs_data[i_plot].plot(T_smooth, y_center, color=colors_fits[0], label='Labfit {:.3f}(T/T)$^{{{:.2f}}}$'.format(base,n))
                axs_data[i_plot].fill_between(T_smooth, y_min, y_max, color=colors_fits[0], alpha=.2)
            
                df_sceg.loc[feat, 'RMS_labfit_'+data.index[0][:-4]] = np.sqrt(np.sum((SPL(T_conditions, base, n) - data)**2))
            

            
            if data_labfit[i_plot][:2] != 'sd': 
    
                # fit the data using DPL (weighted by Labfit uncertainties)
                solving=True
                i_solve=-1
                p0s = [[base, n, base/10, n/10], [base, -1*n, base/10, n/10], [base, -1*n, 0, 0]]
                while solving:
                    i_solve+=1
                    try: 
                        p_DPL, cov = fit(DPL, T_conditions, data, p0=p0s[i_solve], maxfev=5000, sigma=uc_data.to_numpy(float))
                        p_err = np.sqrt(np.diag(cov))
    
                        solving=False # you solved it!
                        axs_data[i_plot].plot(T_smooth, DPL(T_smooth, *p_DPL), color=colors_fits[2], label='DPL {:.3f}(T/T)$^{{{:.2f}}}$+{:.3f}(T/T)$^{{{:.2f}}}$'.format(p_DPL[0],p_DPL[1],p_DPL[2],p_DPL[3]))
                        
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
                            # axs[i_1,i_2].fill_between(T_smooth, y_min, y_max, color=colors_fits[2], alpha=.2)
                        
                        df_sceg.loc[feat, 'RMS_DPL_'+data.index[0][:-4]] = np.sqrt(np.sum((DPL(T_conditions, *p_DPL) - data)**2))
                        df_sceg.loc[feat, 'condition_DPL_'+data.index[0][:-4]] = np.linalg.cond(cov)
                        
                    except: print('could not solve DPL for {}'.format(feat))    
                    if i_solve == len(p0s)-1: solving=False # you didn't solve it, but it's time to move on
                
            # housekeeping to make the plots look nice
            axs_data[i_plot].set_ylabel(name_plot[i_plot])
            
            # DPL makes axis zoom out too much. specify zoom. 
            y_min = min(data-uc_data.to_numpy(float))
            y_max = max(data+uc_data.to_numpy(float))
            axs_data[i_plot].set_ylim(y_min-0.1*np.abs(y_min), y_max+0.1*np.abs(y_max))
            
            axs_data[i_plot].legend()
            
            axs_data[i_plot].tick_params(axis='both', which='both', direction='in', top=True, bottom=True, left=True, right=True)
            axs_data[i_plot].xaxis.set_minor_locator(AutoMinorLocator(5))
            axs_data[i_plot].yaxis.set_minor_locator(AutoMinorLocator(5))
            
            # set up x-axis for all plots
            if i_plot == 2: 
                
                axs_data[i_plot].set_xlabel('Temperature (K)')
                axs_data[i_plot].set_xticks(np.arange(300, 1301, 200))
                axs_data[i_plot].set_xlim(200, 1400)
                
            elif i_plot in [0,1]:
                
                axs_data[i_plot].set_xticks(np.arange(300, 1301, 200))
                axs_data[i_plot].set_xticklabels([])
                axs_data[i_plot].set_xlim(200, 1400)
                

        # plot transimission data
        linewidth = 1
        colors_trans = ['k', '#0028ff','#0080af','#117d11','#be961e','#ff0000',     '#e6ab02', '#fee9ac']
        nu_center = df_feat.nu
        

        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.015, wspace=0.005) # rows, columns
        
        for i, which_file in enumerate(which_files): 
            
            axs_trans[0].plot(wvn[i],trans[i], color=colors_trans[i], label=which_file, 
                      linewidth=linewidth)
            axs_trans[0].legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10)
            
            
            # if adjust: 
            #     if i == 0: res_HT_og = res_HT.copy()
            #     if i == 1:  res_HT[i] += 0.002
            #     if i == 2:  res_HT[i] += 0.003
                
            axs_trans[1].plot(wvn[i],res_HT[i], color=colors_trans[i], label=which_file, 
                      linewidth=linewidth)
            
            axs_trans[2].plot(wvn[i],res_updated[i], color=colors_trans[i], label=which_file, 
                      linewidth=linewidth)
            
            
            
            
        # set axis
        axs_trans[0].set_xlim(nu_span[features_plot.index(feat)][0], nu_span[features_plot.index(feat)][1])
        
        
        asdfsddddddd
        
        axs_trans[0].set_ylim(y_lim_top)
        axs_trans[1].set_ylim(y_lim_bottom)
        axs_trans[2].set_ylim(y_lim_bottom)
        
        
        
        #  remove x label on upper plots (mostly covered)
        plt.setp(ax00.get_xticklabels(), visible=False) 
        plt.setp(ax10.get_xticklabels(), visible=False) 
        
        
        
        # add ticks and minor ticks all over
        ax00.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
        ax00.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax00.yaxis.set_minor_locator(AutoMinorLocator(10))
        
        ax10.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
        ax10.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax10.yaxis.set_minor_locator(AutoMinorLocator(10))
        
        ax20.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
        ax20.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax20.yaxis.set_minor_locator(AutoMinorLocator(10))
        
        
        # labels
        ax20.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
        
        ax00.set_ylabel('Measured\nTransmission')
        ax10.set_ylabel('Meas-\nHITRAN')
        ax20.set_ylabel('Meas-\nT.W.')



































                
                
                
        
        plt.tight_layout()
                
        plt.savefig(os.path.abspath('')+r'\plots\DPL with transmission {}.png'.format(feat), bbox_inches='tight',pad_inches = 0.1)
        
        
        ssssssssssssssssssssssssssss
        
        plt.close()
                 




