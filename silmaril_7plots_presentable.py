r'''

silmaril7_plots

plots data after processing it into a pckl'd file in silmaril6


r'''



import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import linelist_conversions as db
import fig_format
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import labfithelp as lab

import os
import pickle
import pldspectrapy as pld
import td_support as td

import scipy.stats as ss
from sklearn.metrics import r2_score

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()



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
cutoff_s296 = 1E-24 

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower'] # note that SD_self is really SD_air 

d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'


HT_errors = {'0': '0 (unreported)', 
             '1': '1 (default)', 
             '2': '2 (average)', 
             '3': '3 (over 20%)', 
             '4': '4 (10-20%)', 
             '5': '5 (5-10%)', 
             '6': '6 (2-5%)', 
             '7': '7 (1-2%)', 
             '8': '8 (under 1%)'}

# HT_errors_nu = {'0': '0 (over 1)', 
#                 '1':             '1 (0.1-1)', 
#                 '2':            '2 (0.01-0.1)', 
#                 '3':           '3 (0.001-0.01)', 
#                 '4':          '4 (0.0001-0.001)', 
#                 '5':         '5 (0.00001-0.0001)', 
#                 '6':        '6 (0.000001-0.00001)', 
#                 '7':       '7 (0.0000001-0.000001)', 
#                 '8':      '8 (0.00000001-0.0000001)'}

HT_errors_nu = {'0': '0 (>10$^{0}$ cm$^{-1}$)', 
                '1': '1 (<10$^{0}$ cm$^{-1}$)', 
                '2': '2 (<10$^{-1}$ cm$^{-1}$)', 
                '3': '3 (<10$^{-2}$ cm$^{-1}$)', 
                '4': '4 (<10$^{-3}$ cm$^{-1}$)', 
                '5': '5 (<10$^{-4}$ cm$^{-1}$)', 
                '6': '6 (<10$^{-5}$ cm$^{-1}$)', 
                '7': '7 (<10$^{-6}$ cm$^{-1}$)', 
                '8': '8 (<10$^{-7}$ cm$^{-1}$)', 
                '9': '9 (<10$^{-8}$ cm$^{-1}$)'}

HT_errors_nu_val = {'0': 1e0, 
                    '1': 1e0, 
                    '2': 1e-1, 
                    '3': 1e-2, 
                    '4': 1e-3, 
                    '5': 1e-4, 
                    '6': 1e-5, 
                    '7': 1e-6, 
                    '8': 1e-7, 
                    '9': 1e-8}



# %% other stuff to put at the top here


markers = ['1','2','3','+','x', '.', '.', '.']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']

colors = ['dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'purple', 'moccasin']
colors_grad = ['firebrick','orangered','goldenrod','forestgreen','teal','royalblue','mediumpurple', 'darkmagenta']



# %% read in results, re-write quantum assignments in a way that is useful

if d_type == 'pure': f = open(os.path.join(d_sceg,'df_sceg_pure.pckl'), 'rb')
elif d_type == 'air': f = open(os.path.join(d_sceg,'df_sceg_air.pckl'), 'rb')
[df_sceg, df_HT2020, df_HT2020_HT, df_HT2016_HT, df_paul] = pickle.load(f)
f.close()

df_sceg.loc[df_sceg.uc_nu==0, 'uc_nu'] = 0.0015 
df_sceg.loc[df_sceg.uc_sw==0, 'uc_sw'] = df_sceg[df_sceg.uc_nu==0].uc_sw * 0.09
df_sceg.loc[df_sceg.uc_gamma_self==0, 'uc_gamma_self'] = 0.1
df_sceg.loc[df_sceg.uc_n_self==0, 'uc_n_self'] = 0.13
df_sceg.loc[df_sceg.uc_sd_self==0, 'uc_sd_self'] = 0.1 

df_sceg['uc_nu_stat'] = df_sceg.uc_nu.copy()
df_sceg['uc_sw_stat'] = df_sceg.uc_sw.copy()
df_sceg['uc_elower_stat'] = df_sceg.uc_elower.copy()

df_sceg['uc_gamma_self_stat'] = df_sceg.uc_gamma_self.copy()
df_sceg['uc_n_self_stat'] = df_sceg.uc_n_self.copy()
df_sceg['uc_sd_self_stat'] = df_sceg.uc_sd_self.copy()

df_sceg['uc_delta_self_stat'] = df_sceg.uc_delta_self.copy()
df_sceg['uc_n_delta_self_stat'] = df_sceg.uc_n_delta_self.copy()


which = (df_sceg.uc_nu>-0.5)
df_sceg.loc[which, 'uc_nu'] = np.sqrt(df_sceg[which].uc_nu_stat**2 + 
                               (1.7E-4)**2)
which = (df_sceg.uc_sw>-0.5)
df_sceg.loc[which, 'uc_sw'] = np.sqrt((df_sceg[which].uc_sw_stat/df_sceg[which].sw)**2 + 
                               (0.0086)**2) * df_sceg[which].sw

which = (df_sceg.uc_elower>-0.5)
df_sceg['uc_elower'] = np.sqrt((df_sceg[which].uc_elower_stat/df_sceg[which].elower)**2 + 
                               (0.0081)**2) * df_sceg[which].elower

which = (df_sceg.uc_gamma_self>-0.5)
df_sceg.loc[which, 'uc_gamma_self'] = np.sqrt((df_sceg[which].uc_gamma_self_stat/df_sceg[which].gamma_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2) * df_sceg[which].gamma_self
which = (df_sceg.uc_n_self>-0.5)
df_sceg.loc[which, 'uc_n_self'] = np.sqrt((df_sceg[which].uc_n_self_stat/df_sceg[which].n_self)**2 + 
                                   (0.9645*df_sceg[which].uc_gamma_self/df_sceg[which].gamma_self)**2) * df_sceg[which].n_self
which = (df_sceg.uc_sd_self>-0.5)
df_sceg.loc[which, 'uc_sd_self'] = np.sqrt((df_sceg[which].uc_sd_self_stat/df_sceg[which].sd_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2 + 0.03**2) * df_sceg[which].sd_self

which = (df_sceg.uc_delta_self>-0.5)
df_sceg.loc[which, 'uc_delta_self'] = np.sqrt((df_sceg[which].uc_delta_self_stat/df_sceg[which].delta_self)**2 + 
                                       (0.0027)**2 + (0.0081)**2 + #) * df_sceg[which].delta_self
                                       (1.7E-4 / (0.021*df_sceg[which].delta_self))**2) * df_sceg[which].delta_self

which = (df_sceg.uc_n_delta_self>-0.5)
df_sceg.loc[which, 'uc_n_delta_self'] = np.sqrt((df_sceg[which].uc_n_delta_self_stat/df_sceg[which].n_delta_self)**2 + 
                                   (0.9645*df_sceg[which].uc_delta_self/df_sceg[which].delta_self)**2) * df_sceg[which].n_delta_self


# f = open(os.path.join(d_sceg,'df_sceg_all.pckl'), 'rb')
# [df_sceg] = pickle.load(f)
# f.close()

# f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
# [T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_HT_pure] = pickle.load(f)
# f.close()

df_sceg_align, df_HT2020_align = df_sceg.align(df_HT2020, join='inner', axis=0)
df_sceg_align2, df_HT2020_HT_align = df_sceg_align.align(df_HT2020_HT, join='inner', axis=0)


if not df_sceg_align.equals(df_sceg_align2): throw = error2please # these should be the same dataframe if everything lines up





df_sceg['quanta_index'] = df_sceg.quanta.replace(r'\s+', ' ', regex=True)
df_sceg['iso_quanta_index'] = df_sceg.local_iso_id + df_sceg.quanta_index

df_HT2020_HT['iso_quanta_index'] = df_HT2020_HT.local_iso_id.astype(str) + ' ' + df_HT2020_HT.quanta_index
df_HT2016_HT['iso_quanta_index'] = df_HT2016_HT.local_iso_id.astype(str) + ' ' + df_HT2016_HT.quanta_index

df_all_sw = df_HT2020_HT.drop(columns = ['molec_id', 'local_iso_id' ,'gamma_air', 'gamma_self', 'n_air', 'delta_air', 'quanta', 'other', 'quanta_index'])

# merge in sceg
df_all_sw = pd.merge(df_all_sw, df_sceg[['sw','uc_sw','elower']], how='inner', left_index=True, right_index=True, suffixes=('_2020', '_sceg'))
df_all_sw = df_all_sw.rename(columns={'uc_sw':'uc_sw_sceg'})

# merge in 2016
df_all_sw = pd.merge(df_all_sw, df_HT2016_HT[['sw', 'iso_quanta_index', 'iref']], on='iso_quanta_index', how='inner', suffixes=('_2020', '_2016'))
df_all_sw = df_all_sw.rename(columns={'sw':'sw_2016'})

df_all_sw['sw_ref_2020'] = df_all_sw.iref_2020.str[2:4]
df_all_sw['sw_ref_2016'] = df_all_sw.iref_2016.str[2:4]


please = stopherefdf


# %% percent change SW with E lower


markers = ['v', 'o', '>','s','^','d', '<']
linestyles = ['', 'solid', (5, (10, 3)), 'dashdot', 'dashed','','']
# colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']
colors = ['k','#d95f02','#514c8e','#f5a9d0', '#131f06','k']

buffer = 1.3

df_plot = df_sceg_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])
df_plot_og = df_HT2020_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])

df_plot_ht = df_HT2020_HT_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])
sw_error = df_plot_ht.ierr.str[1]
sw_ref = df_plot_ht.iref.str[2:4]
sw_ref_dict = {}


label_x = 'Line Strength, S$_{296}$ (This Work) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Percent Change in Line Strength, S$_{296}$ \n 100% (This Work - HITRAN) / HITRAN' # ' \n [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_y'] = (df_plot.sw - df_plot_og.sw) / df_plot_og.sw * 100
plot_y = df_plot['plot_y']


plot_y_unc = df_plot.uc_sw
label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_c = df_plot.elower

plt.figure(figsize=(14.4, 5)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)

plt.errorbar(plot_x,plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)


for i, ierr in enumerate(np.sort(sw_error.unique())): 
        
    label = HT_errors[ierr]
    
    which = (sw_error == ierr) #  & (sw_ref == '80')

    sw_ref_dict[ierr] = sw_ref[which] #&(df_plot.local_iso_id == 1)]
        
    delta_avg = np.mean(plot_y[which])
    delta_avg_abs = np.mean(abs(plot_y[which]))
    delta_std_abs = np.std(abs(plot_y[which]))
    
    if ierr == '3': MAD = ' MAD = {:.0f}±{:.0f}%'.format(delta_avg_abs,delta_std_abs)   
    elif ierr == '4': MAD = '    MAD = {:.0f}±{:.0f}%'.format(delta_avg_abs,delta_std_abs)   
    elif ierr == '5': MAD = '      MAD = {:.0f}±{:.0f}%'.format(delta_avg_abs,delta_std_abs)   
    elif ierr == '6':  MAD = '        MAD =   {:.0f}±{:.0f}%'.format(delta_avg_abs,delta_std_abs)   
    else:  MAD = '        MAD = {:.0f}±{:.0f}%'.format(delta_avg_abs,delta_std_abs)   

    sc = plt.scatter(plot_x[which], plot_y[which], marker=markers[i], 
                     c=plot_c[which], cmap='viridis', zorder=2, 
                     label=label+MAD, vmin=0, vmax=6000)
    df_plot.sort_values(by=['sw'], inplace=True)
    
    
    if ierr != '3': 
        within_HT = plot_y[which].abs()
        within_HT = len(within_HT[within_HT < float(HT_errors[ierr].split('-')[-1].split('%')[0])])
    else: within_HT = 'N/A'
        
    print(' {} total, {} within uncertainty'.format(len(plot_x[which]), within_HT))
    
    print('       {}       {} ± {}'.format(delta_avg, delta_avg_abs, delta_std_abs))
    
    # print('{}    {}   {}   {}   {} ± {}     {}  {}'.format(ierr, len(plot_x[which]), within_HT, delta_avg, delta_avg_abs, delta_std_abs, 
    #                                             df_plot[which].elower.mean(), len(df_plot[which][df_plot.local_iso_id != '1'])))

plt.legend(edgecolor='k', framealpha=1, loc='upper left', fontsize=12)

ax = plt.gca()
legend = ax.get_legend()
legend_dict = {handle.get_label(): handle for handle in legend.legendHandles}

legend_dict_keys = list(legend_dict.keys())

for i, ierr in enumerate(np.sort(sw_error.unique())): 
    
    if ierr not in ['7', '6', '5', '3']: 
        
        plt.hlines(float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)
        plt.hlines(-float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)
        
    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors[ierr]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i])
    df_plot.sort_values(by=['sw'], inplace=True)




plt.xlim(min(plot_x)/buffer, max(plot_x)*buffer)
plt.xscale('log')

cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

ax.minorticks_on()

plt.xlim(2.1e-31, 2.5e-20)
plt.ylim(-150, 1999)

plt.xticks(10**np.arange(-30, -19, 1.0))


# ------------ plot inset for HITRAN 2020 ------------

ax_ins = inset_axes(ax, width='48%', height='30%', loc='center right', bbox_to_anchor=(0,-0.06,1,1), bbox_transform=ax.transAxes)

for i, ierr in enumerate(np.sort(sw_error.unique())): 
       
    ax_ins.scatter(plot_x[sw_error == ierr], plot_y[sw_error == ierr], marker=markers[i], 
                     c=plot_c[sw_error == ierr], cmap='viridis', zorder=2, 
                     label=HT_errors[ierr])
    
    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors[ierr]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i])
    df_plot.sort_values(by=['sw'], inplace=True)
    
    if ierr not in ['3']: 
        plt.hlines(float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)
        plt.hlines(-float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)
    
   
patch, pp1,pp2 = mark_inset(ax, ax_ins, loc1=1, loc2=2, fc='none', ec='k', zorder=10)
pp1.loc1 = 3
pp2.loc1 = 4

plt.xlim(2e-24, 2.25e-20)
plt.ylim(-12, 12)

plt.xscale('log')

ax_ins.text(.6, .13, "This Work vs HITRAN(2020)", fontsize=12, transform=ax_ins.transAxes) # fontweight="bold",


# ------------ plot inset for HITRAN 2016 ------------

df_plot = df_all_sw[df_all_sw.uc_sw_sceg > -1].sort_values(by=['elower_sceg'])
                                                                 
df_plot['plot_x'] = df_plot.sw_sceg
plot_x = df_plot['plot_x']

df_plot['plot_y'] = (df_plot.sw_sceg - df_plot.sw_2016) / df_plot.sw_2016 *100
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_sw_sceg
label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_c = df_plot.elower_sceg

sw_error = df_plot.ierr.str[1]

df_plot.ierr = df_plot.ierr.str[1]

ax_ins = inset_axes(ax, width='48%', height='30%', loc='upper right', bbox_to_anchor=(0,0,1,1), bbox_transform=ax.transAxes)

for i, ierr in enumerate(np.sort(sw_error.unique())): 
       
    ax_ins.scatter(plot_x[sw_error == ierr], plot_y[sw_error == ierr], marker=markers[i], 
                     c=plot_c[sw_error == ierr], cmap='viridis', zorder=2, 
                     label=HT_errors[ierr])

    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors[ierr]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i])
    df_plot.sort_values(by=['sw_2016'], inplace=True)
    
    if ierr not in ['3']: 
        plt.hlines(float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)
        plt.hlines(-float(HT_errors[ierr].split('-')[-1].split('%')[0]),min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i], linewidth=2)

plt.xlim(2e-24, 2.25e-20)
plt.ylim(-12, 12)

plt.xscale('log')


ax_ins.text(.6, .13, "This Work vs HITRAN2016", fontsize=12, transform=ax_ins.transAxes) # fontweight="bold",


plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 SW.svg',bbox_inches='tight')



# %% change in line center with E lower




markers = ['v', 'o', '>','s','^','d', '<']
linestyles = ['', 'solid', (5, (10, 3)), 'dashdot', 'dashed','','']
# colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']
colors = ['k', '#4c7c17','#d95f02','#514c8e','#f5a9d0', 'k','k', '#1b9e77']


buffer = 1.3

df_plot = df_sceg_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
df_plot_og = df_HT2020_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])

df_plot_ht = df_HT2020_HT_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
nu_error = df_plot_ht.ierr.str[0]
nu_ref = df_plot_ht.iref.str[0:2]
nu_ref_dict = {}

label_x = 'Line Strength, S$_{296}$ (This Work) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Difference in Line Position, $\Delta\\sigma_{0}$ \n (This Work - HITRAN) [cm$^{-1}$]'
df_plot['plot_y'] = df_plot.nu - df_plot_og.nu
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_nu
label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_c = df_plot.elower


plt.figure(figsize=(14.4, 5)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


for i_err, err in enumerate(np.sort(nu_error.unique())):  
    
    which = (nu_error == err)  # &(nu_ref == '80')  &(nu_error == '6')
    
    nu_ref_dict[err] = nu_ref[which]
    
    delta_avg = np.mean(plot_y[which])
    delta_avg_abs = np.mean(abs(plot_y[which]))
    delta_std_abs = np.std(abs(plot_y[which]))
    
    plt.errorbar(plot_x[which],plot_y[which], yerr=plot_y_unc[which], color='k', ls='none', zorder=1)
    
    if err == '1': MAD = '   MAD = {:.4f}±{:.4f}'.format(delta_avg_abs,delta_std_abs) 
    elif err == '6': MAD = ' MAD = {:.4f} (1 feat.)'.format(delta_avg_abs)
    else: MAD = ' MAD = {:.4f}±{:.4f}'.format(delta_avg_abs,delta_std_abs)
    
    sc = plt.scatter(plot_x[which], plot_y[which], marker=markers[i_err], 
                     c=plot_c[which], cmap='viridis', zorder=2, 
                     label=HT_errors_nu[err] + MAD, vmin=0, vmax=6000)
    df_plot.sort_values(by=['sw'], inplace=True)
    
    within_HT = plot_y[which].abs()
    within_HT = len(within_HT[within_HT < HT_errors_nu_val[err]])
    
    print('{} -  {} total, {} within uncertainty ({}%)'.format(err,len(plot_x[which]), within_HT, 100*within_HT/(len(plot_x[which])-1e-15)))
    
    print('       {}       {} ± {}'.format(delta_avg, delta_avg_abs, delta_std_abs))

    print(np.mean(plot_c[which]))
    print(np.std(plot_c[which]))


plt.plot([1e-29,1e-29], [0,0], color=colors[-1], label='DCS unc. (±1.7×10$^{-4}$)', linewidth=6)


handles, labels = plt.gca().get_legend_handles_labels()
order = [1,2,3,4,5,6,7,0]

plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
           loc='upper right', ncol=2, edgecolor='k', framealpha=1, labelspacing=0, 
           fontsize=12)

ax = plt.gca()
legend = ax.get_legend()
legend_dict = {handle.get_label(): handle for handle in legend.legendHandles}

legend_dict_keys = list(legend_dict.keys())

ax.minorticks_on()

plt.xlim(2.1e-31, 2.5e-20)
plt.ylim(-0.19, 0.14)

plt.xscale('log')

cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

plt.xticks(10**np.arange(-30, -19, 1.0))

# lines for HITRAN uncertainties

for i_err, err in enumerate(np.sort(nu_error.unique())): 
       
    if err not in ['1', '6', '7']: 
        
        if err == '2': # don't extend as far to avoid inset
            plt.hlines(HT_errors_nu_val[err],min(plot_x), 10e-28,
                   linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)
            plt.hlines(-HT_errors_nu_val[err],min(plot_x), 5e-28,
                       linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)
        else: 
            plt.hlines(HT_errors_nu_val[err],min(plot_x), max(plot_x),
                   linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)
            plt.hlines(-HT_errors_nu_val[err],min(plot_x), max(plot_x),
                       linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)
    
    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors_nu[err]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i_err])
    df_plot.sort_values(by=['sw'], inplace=True)



# plot inset 

ax_ins = inset_axes(ax, width='64%', height='35%', loc='lower right', bbox_to_anchor=(0,0.07,1,1), bbox_transform=ax.transAxes)

for ierr, err in enumerate(np.sort(nu_error.unique())): 
   
    ax_ins.errorbar(plot_x[nu_error == err],plot_y[nu_error == err], yerr=plot_y_unc[nu_error == err], 
                    color='k', ls='none', zorder=1) 
   
    ax_ins.scatter(plot_x[nu_error == err], plot_y[nu_error == err], marker=markers[i_err], 
                     c=plot_c[nu_error == err], cmap='viridis', zorder=2, 
                     label=HT_errors_nu[err])
    df_plot.sort_values(by=['sw'], inplace=True)
    
    
    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors_nu[err]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i_err])

        
for i_err, err in enumerate(np.sort(nu_error.unique())): 
       
    if err not in ['1', '6', '7']: 
        
        plt.hlines(HT_errors_nu_val[err],min(plot_x), max(plot_x),
                   linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)
        plt.hlines(-HT_errors_nu_val[err],min(plot_x), max(plot_x),
                   linestyles=linestyles[i_err], color=colors[i_err], linewidth=2)

    
    legend_line = legend_dict_keys[np.where([key.startswith(HT_errors_nu[err]) for key in legend_dict])[0].tolist()[0]]
    legend_dict[legend_line].set_color(colors[i_err])
    df_plot.sort_values(by=['sw'], inplace=True)


ax_ins.axhspan(-1.7e-4, 1.7e-4, alpha=1, color=colors[-1], zorder=0)




patch, pp1,pp2 = mark_inset(ax, ax_ins, loc1=1, loc2=2, fc='none', ec='k', zorder=10)
pp1.loc1 = 1
pp2.loc1 = 2

plt.xscale('log')

plt.xlim(2e-26, 2.5e-20)
plt.ylim(-0.0019, 0.0019)

# need to reduce spacing between lines with added cm-1 before saving to file

# plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 NU.svg',bbox_inches='tight')


# %% feature widths - J plot


plot_which_y = 'gamma_self'
label_y = 'Self-Width, γ$_{self}$ [cm$^{-1}$/atm]'

# 
# label_y = 'Self-Width Temperature Exponent, n$_{self}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)] #&(df_sceg['uc_n_self'] > -1)] # floating all width parameters

# df_plot_ht = df_HT2020_HT_align[(df_sceg['uc_gamma_self'] > -1)]
# g_error = df_plot_ht.ierr.str[3]
# g_ref = df_plot_ht.iref.str[6:8]
# g_ref_dict = {}


plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)



# plot_x = df_plot[['Kap', 'Kapp']].max(axis=1) + 0.1*(df_plot[['Jp', 'Jpp']].max(axis=1) - df_plot[['Kap', 'Kapp']].max(axis=1))
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]

 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=5000)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
j_HT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
g_HT = [0.50361, 0.47957, 0.45633, 0.43388, 0.41221, 0.39129, 0.37113, 0.3517, 0.333, 0.31501, 
        0.29773, 0.28113, 0.26521, 0.24996, 0.23536, 0.2214, 0.20806, 0.19534, 0.18323, 0.1717, 
        0.16076, 0.15038, 0.14056, 0.13128, 0.12252, 0.11429]

plt.plot(j_HT,g_HT, colors[0], label='HITRAN/HITEMP', linewidth=4)
# plt.plot([0, 25], [.484, 0.484-0.018*25], colors[1], label='This Work (0.484-0.018J")',
#          linewidth=4, linestyle='dashed')
plt.plot(j_HT, 0.485*np.exp(-0.0633*np.array(j_HT)) + 0.04, colors[1], label='This Work (0.485 exp[-0.0633J"] + 0.04)',
         linewidth=4, linestyle='dashed')



plt.legend(loc='upper right', ncol=1, edgecolor='k', framealpha=1, labelspacing=0.5)

cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

ax = plt.gca()
ax.minorticks_on()

plt.xlim(-.9,24.9)
plt.ylim(-0.04,1.33)

plt.xticks(np.arange(0, 25, 2.0))

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width Linda.svg',bbox_inches='tight')

# %% feature temperature dependence of widths - J" plot


plot_which_y = 'n_self'
label_y = 'Self-Width Temperature Exponent, n$_{γ,self}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)] # floating all width parameters

plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)



# plot_x = df_plot[['Kap', 'Kapp']].max(axis=1) + 0.1*(df_plot[['Jp', 'Jpp']].max(axis=1) - df_plot[['Kap', 'Kapp']].max(axis=1))
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]


 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=3000)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
p = np.polyfit(plot_x, plot_y, 1)
plot_x_sparse = [0, 18]
plot_y_fit = np.poly1d(p)(plot_x_sparse)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

plt.plot([0,18], [0.997, 0.997-0.068*18], colors[1], label='This Work (0.997-0.068J")',
         linewidth=4, linestyle='dashed')

plt.legend(loc='lower left', ncol=2, edgecolor='k', framealpha=1, labelspacing=0)


cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

plt.xlim(-.9,19.9)
plt.ylim(-0.86,1.3)
plt.xticks(np.arange(0, 19, 2.0))

ax = plt.gca()
ax.minorticks_on()



plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 n width Linda.svg',bbox_inches='tight')





# %% feature widths vs HITRAN

plot_which = 'n_self'
label_y = 'n$_{γ,self}$, This Work [cm$^{-1}$/atm]'
label_x = 'n$_{γ,air}$, HITRAN [cm$^{-1}$/atm]'

label_c = 'Angular Momentum of Ground State, J"'

which = (df_sceg['uc_n_self'] > -1) # &(df_sceg['uc_n_self'] > -1)

df_plot = df_sceg_align[which].sort_values(by=['Jpp'])
df_HT2020_align['Jpp'] = df_sceg_align.Jpp
df_plot_HT = df_HT2020_align[which].sort_values(by=['Jpp'])

# df_plot['gamma_self'] = df_HT2020_align.gamma_self
# df_plot['n_self'] = df_HT2020_align.n_self

# df_plot_ht = df_HT2020_HT_align[(df_sceg['uc_gamma_self'] > -1)]
# g_error = df_plot_ht.ierr.str[3]
# g_ref = df_plot_ht.iref.str[6:8]
# g_ref_dict = {}

# df_plot_HT = df_plot_HT[g_ref == '71']


plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


plot_x = df_plot_HT['n_air']
plot_y = df_plot[plot_which]
plot_c = df_plot_HT.Jpp



 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
if plot_logx: 
    plt.xscale('log')
    
# plt.legend()


cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

line_ = [0, 1]

# plt.plot(line_,line_,'k',linewidth=2)

p = np.polyfit(plot_x, plot_y, 1)
plot_y_fit = np.poly1d(p)(line_)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

r2 = r_value**2

plt.plot(line_, plot_y_fit, colors[1], label='This Work  ({}n$_{}${})'.format(str(slope)[:4],'HT',str(intercept)[:5]),
          linewidth=4, linestyle='dashed')

plt.legend(loc='lower right', edgecolor='k', framealpha=1)

plt.show()

# plt.ylim(-.59,1.24)
# plt.xlim(0.05,0.6)

# plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width HT.svg',bbox_inches='tight')


mad = np.mean(np.abs(plot_y-np.poly1d(p)(plot_x)))
rms = np.sqrt(np.sum((plot_y-np.poly1d(p)(plot_x))**2)/ len(plot_y))
r2 = r2_score(plot_y, np.poly1d(p)(plot_x))


print(rms)


# %% feature widths vs temp dependence (matching Paul's style)

colors = ['#d95f02','k']

#-----------------------
plot_which_y = 'n_self'
label_y = 'Self-Width Temperature Exponent, n$_{γ,self}$'

plot_which_x = 'gamma_self'
label_x = 'Self-Width, γ$_{self}$ [cm$^{-1}$/atm]'

label_c = 'Angular Momentum of Ground State, J"'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)].sort_values(by=['Jpp']).sort_values(by=['Jpp'])

plot_unc_y_bool = True
plot_unc_x_bool = True

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)



plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]
plot_c = df_plot.Jpp


 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
if plot_logx: 
    plt.xscale('log')
    
# plt.legend()


cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

p = np.polyfit(plot_x[(plot_x>0.2)], plot_y[(plot_x>0.2)], 1)
plot_x_sparse = [0.2, 0.5]
plot_y_fit = np.poly1d(p)(plot_x_sparse)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

plt.plot(plot_x_sparse,[1.5344*.2+0.0502, 1.5344*.5+0.0502], colors[0], label='Schroeder  (0.05+1.53γ)', linewidth=4)
plt.plot(plot_x_sparse, plot_y_fit, colors[1], label='This Work ({}+{}γ)'.format(str(intercept)[:5], str(slope)[:4]),
         linewidth=4, linestyle='dashed')

plt.legend(loc='lower right', edgecolor='k', framealpha=1)


plt.show()

plt.ylim(-.69,1.29)
plt.xlim(0.05,0.57)

plt.xticks(np.arange(0.1, 0.6, 0.1))


ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width Paul.svg',bbox_inches='tight')



# %% speed dependence of feature widths


colors = ['#d95f02','#514c8e']

#-----------------------
plot_which_y = 'sd_self'
label_y = 'Speed Dependence of the Self-Width, a$_{w}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)&(df_sceg['uc_sd_self'] > -1)].sort_values(by=['Jpp']) # all width (with SD)

plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False

plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


# plot_x = df_plot[plot_which_x]
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]

sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=3000)

if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

polyfit = np.polyfit(plot_x, plot_y, 0, full=True)
p = polyfit[0]
fit_stats = polyfit[1:]
plot_x_sparse = [0, 19]
plot_y_fit = np.poly1d(p)(plot_x_sparse)


std = np.std(np.poly1d(p)(plot_x) - plot_y)
# r2 = r2_score(plot_y, np.poly1d(p)(plot_x))

plt.plot(plot_x_sparse,[0.125597, 0.125597], colors[0], label='Schroeder Average ({})'.format('0.126'), linewidth=4)
plt.plot(plot_x_sparse, plot_y_fit, color='k', label='This Work Average ({})'.format(str(p[0])[0:5]), linewidth=4, linestyle='dashed')

plt.legend(loc='upper right', edgecolor='k', framealpha=1, fontsize=12)

plt.ylim((-0.04, 0.39))
plt.xlim(-.9,19.9)
plt.xticks(np.arange(0, 19, 2.0))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 SD.svg',bbox_inches='tight')

# %% shift 

markers = ['1','2','3','+','x', '.', '.', '.']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']

#-----------------------
plot_which_y = 'delta_self'
label_y = 'Self-Shift, δ$_{self}$ [cm$^{-1}$/atm]'

plot_which_x = 'elower'
label_x = 'Lower State Energy, E" [cm$^{-1}$]'

colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']
colors = ['dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'purple', 'moccasin']


legend_dict = {'101': ['ν$_{1}$+ν$_{3}$', '#1b9e77'],
               '021': ['2ν$_{2}$+ν$_{3}$','#e6ab02'],
               '111': ['ν$_{1}$+ν$_{2}$+ν$_{3}$←ν$_{2}$','#514c8e'],
               '200': ['2ν$_{1}$', '#d95f02'],
               '120': ['ν$_{1}$+2ν$_{2}$', '#4c7c17'], 
               '002': ['2ν$_{3}$', 'firebrick'], 
               '040': ['4ν$_{2}$', 'darkgreen'],
               '031': ['3ν$_{2}$+ν$_{3}$←ν$_{2}$', 'dodgerblue']}



# plot_which_x = 'm'
# label_x = 'm'

# plot_which_x = 'nu'
# label_x = 'wavenumber'


plot_unc_y_bool = True

plot_labels = False
plot_logx = False

plot_unc_x_bool = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


i=0


df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)] # &(df_sceg['uc_n_delta_self'] > -1)]
plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]

# plt.plot(plot_x, plot_y, '+', color='k', label = 'All (This Work)', linewidth=2)

for vp in df_sceg.vp.unique(): 

    df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg.vp == vp)]
    
    plot_x = df_plot[plot_which_x]
    plot_y = df_plot[plot_which_y]
        
    if len(df_plot) > 0: 
    
        plt.plot(plot_x, plot_y, 'x', color=legend_dict[vp][1], label=legend_dict[vp][0], linewidth=2)
    
        if plot_unc_x_bool: 
            plot_unc_x = df_plot['uc_'+plot_which_x]
            plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color=legend_dict[vp][1], ls='none')
        if plot_unc_y_bool: 
            plot_unc_y = df_plot['uc_'+plot_which_y]
            plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color=legend_dict[vp][1], ls='none')
        
        
    
if plot_logx: 
    plt.xscale('log')
    
plt.legend(loc='lower left', edgecolor='k', framealpha=1, labelspacing=0, fontsize=12, ncol=2)


plt.ylim((-0.63, 0.38))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 shift.svg',bbox_inches='tight')


# %% temperature dependence of the shift (not planning to include this in the final paper)

markers = ['1','2','3','+','x', '.', '.', '.']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']


#-----------------------
plot_which_y = 'n_delta_self'
label_y = 'Self-Shift Temperature Exponent, n$_{δ,self}$'

plot_which_x = 'elower'
label_x = 'Lower State Energy, E" [cm$^{-1}$]'

# -----------------------
# plot_which_y = 'n_delta_self'
# label_y = 'Temp. Dep. of Pressure Shift'

# plot_which_x = 'delta_self'
# label_x = 'Pressure Shift'


legend_dict = {'101': ['ν$_{1}$+ν$_{3}$', '#1b9e77'],
               '021': ['2ν$_{2}$+ν$_{3}$','#e6ab02'],
               '111': ['ν$_{1}$+ν$_{2}$+ν$_{3}$←ν$_{2}$','#514c8e'],
               '200': ['2ν$_{1}$', '#d95f02'],
               '120': ['ν$_{1}$+2ν$_{2}$', '#4c7c17'], 
               '002': ['2ν$_{3}$', 'firebrick'], 
               '040': ['4ν$_{2}$', 'darkgreen'],
               '031': ['3ν$_{2}$+ν$_{3}$←ν$_{2}$', 'dodgerblue']}


plot_unc_y_bool = True

plot_labels = False
plot_logx = False

plot_unc_x_bool = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


i=0


df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)]
plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]

# plt.plot(plot_x, plot_y, '+', color='k', label = 'All (This Work)', linewidth=2)

for vp in df_sceg.vp.unique(): 

    df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg.vp == vp)]
    
    plot_x = df_plot[plot_which_x]
    plot_y = df_plot[plot_which_y]
        
    if len(df_plot) > 0: 
    
        plt.plot(plot_x, plot_y, 'x', color=legend_dict[vp][1], label=legend_dict[vp][0], linewidth=2)
    
        if plot_unc_x_bool: 
            plot_unc_x = df_plot['uc_'+plot_which_x]
            plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color=legend_dict[vp][1], ls='none')
        if plot_unc_y_bool: 
            plot_unc_y = df_plot['uc_'+plot_which_y]
            plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color=legend_dict[vp][1], ls='none')
        
        
    
if plot_logx: 
    plt.xscale('log')
    
plt.legend(loc='lower right', edgecolor='k', framealpha=1, labelspacing=0, fontsize=12)

plt.ylim((-0.89, 5.5))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 n shift.svg',bbox_inches='tight')


# %% table of results (number of fits, vibrational bands, etc.)


props_which = ['uc_nu','uc_sw','uc_gamma_self','uc_n_self','uc_sd_self','uc_delta_self','uc_n_delta_self', 'uc_elower']

count = np.zeros((50,len(props_which)+3))
count_name = []

j = 0

for local_iso_id_iter in df_sceg_align.local_iso_id.unique(): 
    
    df_sceg_iso = df_sceg_align[df_sceg_align.local_iso_id == local_iso_id_iter]
       
    for vp_iter in df_sceg_iso.vp.unique(): 
        
        df_sceg_vp = df_sceg_iso[df_sceg_iso.vp == vp_iter]
        
        for i, prop in enumerate(props_which): 
            
            count[j,i] = len(df_sceg_vp[df_sceg_vp[prop] > 0])
            
        if count[j,:].any(): 
            
            # iso, vp, vpp, J min, J max, 
            
            count_name.append(vp_iter + '-' + df_sceg_vp.vpp.mode()[0])
            
            
            if count_name[j] == '-': count_name[j] = '-2-2-2?'
            
            print(count_name[j])
            
            count[j,-1] = max(df_sceg_vp.Jpp)
            count[j,-2] = min(df_sceg_vp.Jpp)
            
            count[j,-3] = local_iso_id_iter
        
            print(count[j,:])
                        
            j+=1
                
                
            
            
df_sceg_new = df_sceg[df_sceg.index > 100000]

for i, prop in enumerate(props_which): 

    count[-1,i] = len(df_sceg_new[df_sceg_new[prop] > 0])
            


#%% save as csv for publication


df_sceg_save = df_sceg[['molec_id', 'local_iso_id', 'mass', 'quanta',
                        'nu', 'uc_nu', 'uc_nu_stat',
                        'sw', 'uc_sw',  'uc_sw_stat',
                        'elower', 'uc_elower', 'uc_elower_stat',
                        'gamma_self', 'uc_gamma_self', 'uc_gamma_self_stat',
                        'n_self', 'uc_n_self', 'uc_n_self_stat',
                        'sd_self', 'uc_sd_self', 'uc_sd_self_stat',
                        'delta_self', 'uc_delta_self', 'uc_delta_self_stat',
                        'n_delta_self', 'uc_n_delta_self', 'uc_n_delta_self_stat']].copy()

df_sceg_save = df_sceg_save.rename(columns={'uc_nu': 'uc_nu_total',
                             'uc_sw': 'uc_sw_total',
                             'uc_elower': 'uc_elower_total',
                             'uc_gamma_self': 'uc_gamma_self_total',
                             'uc_n_self': 'uc_n_self_total',
                             'uc_sd_self': 'uc_sd_self_total',
                             'uc_delta_self': 'uc_delta_self_total',
                             'uc_n_delta_self': 'uc_n_delta_self_total'})

df_sceg_save['doublet'] = ''

doublets_nu = [[4130,4129],[4885,4884],[5204,5205],[5625,5624],[5820,5819],[5858,5856],[5903,5902],[5930,5929],[5980,5979],[5995,5998],[6145,6144],[6390,6389],[6555,6554],[6606,6605],
                [6690,6692],[6708,6705],[6770,6768],[6806,6805],[6930,6929],[6960,6959],[6987,6986],[6989,6985],[7163,7162],[7385,7384],[7623,7622],[7754,7753],[7806,7803],[7812,7810],
                [7842,7839],[7881,7880],[7885,7884],[7892,7891],[7906,7901],[7955,7956],[7981,7980],[8006,8004],[8011,8010],[8213,8212],[8311,8310],[8460,8457],[8667,8666],[8681,8680],
                [8713,8712],[8775,8774],[8784,8782],[8811,8810],[8812,8813],[8849,8848],[8863,8862],[8874,8873],[8882,8883],[8919,8918],[8962,8963],[8979,8980],[8990,8989],[9092,9091],
                [9150,9149],[9471,9470],[9477,9476],[9548,9547],[9567,9568],[9612,9611],[9638,9637],[9714,9712],[9737,9736],[9781,9780],[9793,9792],[9794,9795],[9856,9855],[9868,9867],
                [9951,9950],[9956,9957],[9958,9959],[9977,9978],[10030,10028],[10045,10042],[10114,10113],[10645,10644],[10691,10690],[10783,10781],[10814,10813],[10861,10860],[10864,10862],
                [10884,10883],[10904,10905],[10920,10919],[10980,10979],[10994,10993],[11025,11024],[11086,11085],[11134,11133],[11251,11250],[11488,11482],[11546,11540],[11622,11623],
                [11624,11621],[11644,11643],[11749,11747],[11772,11770],[11788,11787],[11796,11795],[11834,11833],[11882,11881],[11902,11901],[11941,11940],[11997,11996],[12076,12073],
                [12189,12186],[12315,12313],[12413,12414],[12455,12454],[12590,12589],[12632,12631],[12727,12726],[12906,12905],[12981,12980],[12998,12997],[13111,13110],[13202,13201],
                [13369,13368],[13403,13400],[13448,13444],[13481,13480],[13624,13625],[13674,13670],[13699,13690],[13810,13809],[13866,13865],[13922,13920],[14007,14003],[14025,14026],
                [14156,14151],[14172,14169],[14284,14283],[14585,14584],[14680,14677],[14724,14723],[14770,14769],[14850,14844],[14863,14862],[15128,15126],[15313,15312],[15375,15374],
                [15474,15473],[15548,15547],[15588,15587],[15621,15622],[15660,15659],[15874,15873],[15923,15922],[15974,15969],[16047,16045],[16210,16209],[16267,16266],[16469,16466],
                [16491,16488],[16493,16492],[16601,16599],[16705,16704],[16713,16711],[16743,16742],[16767,16766],[17009,17007],[17056,17055],[17069,17068],[17084,17083],[17111,17110],
                [17118,17117],[17147,17146],[17189,17188],[17199,17198],[17244,17241],[17264,17263],[17291,17289],[17332,17331],[17369,17363],[17387,17386],[17475,17473],[17479,17478],
                [17515,17514],[17585,17572],[17604,17602],[17616,17615],[17697,17696],[17733,17732],[17780,17778],[17801,17800],[17833,17831],[17866,17865],[17881,17878],[17938,17937],
                [17994,17993],[18030,18029],[18082,18081],[18117,18116],[18129,18127],[18155,18154],[18172,18171],[18277,18276],[18366,18360],[18373,18370],[18465,18466],[18523,18522],
                [18703,18702],[18712,18711],[19083,19082],[19128,19127],[19131,19123],[19178,19176],[19358,19353],[19410,19409],[19422,19421],[19511,19510],[19535,19533],[19582,19580],
                [19602,19599],[19613,19609],[19654,19653],[19656,19659],[19682,19680],[19692,19683],[19698,19697],[19732,19731],[19814,19812],[19985,19984],[20007,20006],[20050,20049],
                [20115,20112],[20182,20181],[20272,20271],[20345,20344],[20437,20436],[20465,20464],[20469,20468],[20547,20545],[20579,20577],[20586,20581],[20632,20631],[20667,20666],
                [20725,20724],[20814,20815],[20827,20826],[20874,20873],[20953,20954],[21048,21047],[21208,21207],[21240,21233],[21254,21252],[21291,21289],[21301,21299],[21333,21326],
                [21492,21491],[21559,21546],[21576,21574],[21647,21646],[21670,21668],[21766,21765],[21799,21798],[21829,21828],[21844,21843],[21967,21956],[21971,21970],[21973,21975],
                [22037,22036],[22071,22069],[22075,22074],[22129,22128],[22260,22259],[22278,22275],[22373,22367],[22426,22423],[22498,22493],[22516,22515],[22530,22527],[22547,22549],
                [22574,22573],[22639,22638],[22808,22802],[22934,22933],[22946,22947],[22970,22969],[22981,22982],[23011,23010],[23056,23053],[23257,23244],[23293,23292],[23300,23299],
                [23322,23321],[23621,23622],[23665,23666],[23704,23703],[23763,23762],[23782,23781],[23845,23846],[23918,23917],[23923,23922],[23991,23990],[24018,24017],[24065,24060],
                [24083,24071],[24167,24166],[24172,24171],[24354,24351],[24467,24469],[24495,24494],[24763,24762],[24840,24841],[25095,25092],[25118,25117],[25169,25168],[25262,25261],
                [25341,25340],[25454,25453],[25490,25489],[25544,25545],[25823,25822],[26118,26117],[26440,26439],[26483,26485],[26525,26524],[26623,26622],[26678,26677],[27109,27108],
                [27297,27295],[27367,27366],[27569,27570],[27663,27659],[27774,27769],[27779,27777],[27892,27889],[27976,27975],[27988,27987],[28015,28014],[28114,28113],[28127,28126],
                [28234,28231],[28302,28297],[28308,28307],[28315,28314],[28396,28395],[28442,28441],[28448,28447],[28659,28658],[28723,28722],[28779,28778],[28967,28965],[28986,28985],
                [28996,28995],[29010,29009],[29052,29050],[29245,29244],[29256,29255],[29258,29254],[29777,29776],[29982,29983],[30004,30001],[30011,30009],[30048,30047],[30171,30170],
                [30184,30183],[30317,30316],[30450,30449],[30476,30475],[30731,30728],[30787,30783],[30841,30840],[30915,30912],[30940,30939],[30972,30970],[31012,31004],[31026,31025],
                [31103,31102],[31223,31219],[31298,31297],[31339,31340],[31396,31395],[31418,31420],[31471,31470],[32001,32002],[32029,32026],[32135,32134],[32294,32293],[32341,32340],
                [32394,32393],[32428,32427],[32605,32603],[32612,32613],[32669,32668],[32726,32727],[32742,32740],[32908,32907],[33056,33055],[33117,33118],[33217,33215],[33301,33298],
                [33358,33357],[33442,33441],[33478,33477],[33502,33501],[33540,33537],[33696,33695],[33867,33866],[33872,33868],[33885,33884],[33931,33926],[34047,34046],[34130,34129],
                [34195,34193],[34220,34219],[34276,34277],[34291,34290],[34369,34370],[34434,34432],[34435,34429],[34541,34542],[34732,34731],[34771,34770],[34821,34816],[34907,34906],
                [34986,34985],[35033,35027],[35059,35058],[35205,35204],[35235,35236],[35326,35325],[35390,35389],[35431,35430],[35548,35547],[35556,35555],[35581,35579],[35834,35835],
                [35866,35865],[36071,36069],[36075,36074],[36120,36119],[36306,36303],[36318,36317],[36509,36508],[36526,36527],[36662,36661],[36682,36681],[36816,36817],[36844,36845],
                [36953,36952],[36984,36985],[37054,37055],[37088,37087],[37091,37090],[37157,37155],[37192,37191],[37194,37193],[37644,37643],[37990,37991],[37998,37997],[40469,40468]]
doublets_nu = sorted(doublets_nu, key=lambda x: min(x))

doublets_sw = [[3912,3911],[4130,4129],[4885,4884],[5204,5205],[5625,5624],[5639,5638],[5820,5819],[5858,5856],[5876,5875],[5903,5902],[5930,5929],[5980,5979],[5995,5998],[6145,6144],
                [6351,6350],[6390,6389],[6555,6554],[6606,6605],[6690,6692],[6708,6705],[6770,6768],[6806,6805],[6930,6929],[6960,6959],[6987,6986],[6989,6985],[7163,7162],[7385,7384],
                [7623,7622],[7754,7753],[7806,7803],[7812,7810],[7842,7839],[7881,7880],[7885,7884],[7892,7891],[7906,7901],[7955,7956],[7981,7980],[8006,8004],[8011,8010],[8213,8212],
                [8311,8310],[8460,8457],[8474,8477],[8667,8666],[8681,8680],[8713,8712],[8775,8774],[8784,8782],[8811,8810],[8849,8848],[8863,8862],[8874,8873],[8882,8883],[8919,8918],
                [8962,8963],[8979,8980],[8990,8989],[9092,9091],[9150,9149],[9471,9470],[9477,9476],[9548,9547],[9567,9568],[9612,9611],[9638,9637],[9714,9712],[9737,9736],[9781,9780],
                [9793,9792],[9856,9855],[9868,9867],[9951,9950],[9956,9957],[9958,9959],[9977,9978],[10030,10028],[10045,10042],[10114,10113],[10645,10644],[10691,10690],[10783,10781],
                [10814,10813],[10861,10860],[10864,10862],[10884,10883],[10904,10905],[10920,10919],[10980,10979],[10994,10993],[11025,11024],[11086,11085],[11134,11133],[11251,11250],
                [11488,11482],[11546,11540],[11622,11623],[11624,11621],[11644,11643],[11749,11747],[11772,11770],[11788,11787],[11796,11795],[11798,11802],[11834,11833],[11841,11840],
                [11882,11881],[11902,11901],[11941,11940],[11997,11996],[12076,12073],[12189,12186],[12315,12313],[12413,12414],[12455,12454],[12590,12589],[12632,12631],[12727,12726],
                [12906,12905],[12923,12922],[12981,12980],[12998,12997],[13111,13110],[13202,13201],[13369,13368],[13403,13400],[13448,13444],[13481,13480],[13624,13625],[13674,13670],
                [13699,13690],[13783,13782],[13810,13809],[13866,13865],[13922,13920],[14025,14026],[14076,14075],[14156,14151],[14172,14169],[14284,14283],[14680,14677],[14724,14723],
                [14770,14769],[14850,14844],[14863,14862],[15128,15126],[15313,15312],[15375,15374],[15474,15473],[15497,15496],[15548,15547],[15588,15587],[15621,15622],[15660,15659],
                [15830,15829],[15874,15873],[15923,15922],[15974,15969],[16047,16045],[16210,16209],[16267,16266],[16469,16466],[16491,16488],[16493,16492],[16596,16594],[16601,16599],
                [16705,16704],[16713,16711],[16743,16742],[16767,16766],[17009,17007],[17056,17055],[17069,17068],[17084,17083],[17111,17110],[17118,17117],[17147,17146],[17176,17171],
                [17189,17188],[17199,17198],[17244,17241],[17264,17263],[17332,17331],[17369,17363],[17387,17386],[17419,17418],[17475,17473],[17479,17478],[17515,17514],[17531,17534],
                [17585,17572],[17604,17602],[17616,17615],[17733,17732],[17761,17760],[17774,17763],[17780,17778],[17801,17800],[17833,17831],[17866,17865],[17881,17878],[17994,17993],
                [18030,18029],[18082,18081],[18117,18116],[18129,18127],[18155,18154],[18172,18171],[18277,18276],[18293,18292],[18366,18360],[18373,18370],[18465,18466],[18523,18522],
                [18703,18702],[18712,18711],[18964,18962],[19083,19082],[19128,19127],[19131,19123],[19178,19176],[19358,19353],[19410,19409],[19422,19421],[19511,19510],[19535,19533],
                [19582,19580],[19602,19599],[19613,19609],[19654,19653],[19656,19659],[19682,19680],[19692,19683],[19698,19697],[19732,19731],[19814,19812],[19985,19984],[20002,19998],
                [20007,20006],[20050,20049],[20115,20112],[20182,20181],[20272,20271],[20345,20344],[20437,20436],[20465,20464],[20469,20468],[20547,20545],[20579,20577],[20586,20581],
                [20632,20631],[20667,20666],[20725,20724],[20814,20815],[20827,20826],[20874,20873],[20953,20954],[21048,21047],[21208,21207],[21240,21233],[21254,21252],[21291,21289],
                [21301,21299],[21333,21326],[21434,21431],[21492,21491],[21559,21546],[21576,21574],[21647,21646],[21670,21668],[21698,21697],[21766,21765],[21799,21798],[21829,21828],
                [21844,21843],[21971,21970],[21973,21975],[22071,22069],[22075,22074],[22129,22128],[22260,22259],[22278,22275],[22373,22367],[22426,22423],[22498,22493],[22516,22515],
                [22530,22527],[22547,22549],[22574,22573],[22639,22638],[22808,22802],[22934,22933],[22946,22947],[22970,22969],[22981,22982],[23011,23010],[23056,23053],[23257,23244],
                [23293,23292],[23300,23299],[23322,23321],[23621,23622],[23665,23666],[23704,23703],[23763,23762],[23782,23781],[23845,23846],[23918,23917],[23923,23922],[23991,23990],
                [24018,24017],[24065,24060],[24083,24071],[24167,24166],[24172,24171],[24354,24351],[24467,24469],[24495,24494],[24763,24762],[24840,24841],[24947,24946],[25095,25092],
                [25118,25117],[25169,25168],[25262,25261],[25341,25340],[25454,25453],[25490,25489],[25544,25545],[25823,25822],[26118,26117],[26440,26439],[26525,26524],[26623,26622],
                [26678,26677],[27109,27108],[27297,27295],[27367,27366],[27569,27570],[27663,27659],[27774,27769],[27779,27777],[27892,27889],[27976,27975],[27988,27987],[28015,28014],
                [28114,28113],[28127,28126],[28234,28231],[28302,28297],[28308,28307],[28315,28314],[28396,28395],[28442,28441],[28448,28447],[28659,28658],[28723,28722],[28779,28778],
                [28967,28965],[28986,28985],[28996,28995],[29010,29009],[29052,29050],[29245,29244],[29256,29255],[29258,29254],[29623,29622],[29777,29776],[29982,29983],[30004,30001],
                [30011,30009],[30048,30047],[30171,30170],[30184,30183],[30317,30316],[30432,30431],[30450,30449],[30476,30475],[30731,30728],[30787,30783],[30801,30800],[30841,30840],
                [30915,30912],[30940,30939],[30972,30970],[31012,31004],[31026,31025],[31103,31102],[31223,31219],[31298,31297],[31339,31340],[31396,31395],[31418,31420],[32001,32002],
                [32029,32026],[32065,32066],[32135,32134],[32294,32293],[32341,32340],[32356,32355],[32394,32393],[32428,32427],[32605,32603],[32612,32613],[32617,32616],[32669,32668],
                [32726,32727],[32742,32740],[32908,32907],[33056,33055],[33091,33092],[33117,33118],[33217,33215],[33301,33298],[33358,33357],[33442,33441],[33478,33477],[33502,33501],
                [33540,33537],[33696,33695],[33867,33866],[33872,33868],[33885,33884],[33905,33904],[33931,33926],[34047,34046],[34130,34129],[34195,34193],[34220,34219],[34234,34233],
                [34276,34277],[34287,34288],[34291,34290],[34369,34370],[34434,34432],[34435,34429],[34541,34542],[34732,34731],[34771,34770],[34790,34789],[34821,34816],[34878,34879],
                [34907,34906],[34986,34985],[35033,35027],[35059,35058],[35062,35063],[35117,35118],[35205,35204],[35235,35236],[35326,35325],[35390,35389],[35431,35430],[35548,35547],
                [35556,35555],[35568,35567],[35581,35579],[35641,35639],[35834,35835],[35866,35865],[36071,36069],[36075,36074],[36120,36119],[36259,36258],[36306,36303],[36318,36317],
                [36509,36508],[36526,36527],[36662,36661],[36682,36681],[36721,36720],[36816,36817],[36844,36845],[36953,36952],[36984,36985],[37054,37055],[37088,37087],[37091,37090],
                [37157,37155],[37192,37191],[37194,37193],[37644,37643],[37990,37991],[37998,37997],[40469,40468]]
doublets_sw = sorted(doublets_sw, key=lambda x: min(x))

doublets_gamma_self = [[4130,4129],[4885,4884],[5204,5205],[5625,5624],[5820,5819],[5903,5902],[5930,5929],[5980,5979],[5995,5998],[6145,6144],[6606,6605],[6770,6768],[7623,7622],[7754,7753],
                        [7806,7803],[7812,7810],[7842,7839],[7881,7880],[7885,7884],[7906,7901],[7981,7980],[8006,8004],[8011,8010],[8213,8212],[8460,8457],[8681,8680],[8713,8712],[8874,8873],
                        [8882,8883],[8962,8963],[8979,8980],[9150,9149],[9471,9470],[9567,9568],[9612,9611],[9638,9637],[9714,9712],[9793,9792],[9868,9867],[9956,9957],[9958,9959],[9977,9978],
                        [10030,10028],[10114,10113],[10691,10690],[10783,10781],[10814,10813],[10861,10860],[10864,10862],[10904,10905],[10920,10919],[10980,10979],[10994,10993],[11025,11024],
                        [11086,11085],[11134,11133],[11251,11250],[11488,11482],[11546,11540],[11622,11623],[11624,11621],[11644,11643],[11749,11747],[11788,11787],[11841,11840],[11882,11881],
                        [11902,11901],[11997,11996],[12455,12454],[12590,12589],[12632,12631],[12906,12905],[12923,12922],[12998,12997],[13111,13110],[13202,13201],[13369,13368],[13448,13444],
                        [13699,13690],[14025,14026],[14172,14169],[14770,14769],[15128,15126],[15313,15312],[15375,15374],[15474,15473],[15548,15547],[15588,15587],[15621,15622],[15660,15659],
                        [15923,15922],[15974,15969],[16047,16045],[16469,16466],[16743,16742],[16767,16766],[17084,17083],[17118,17117],[17147,17146],[17189,17188],[17199,17198],[17264,17263],
                        [17332,17331],[17369,17363],[17585,17572],[17616,17615],[17780,17778],[17801,17800],[17833,17831],[17866,17865],[17881,17878],[17994,17993],[18030,18029],[18117,18116],
                        [18129,18127],[18155,18154],[18172,18171],[18293,18292],[18366,18360],[18373,18370],[18465,18466],[18523,18522],[18703,18702],[18712,18711],[19128,19127],[19178,19176],
                        [19358,19353],[19410,19409],[19422,19421],[19511,19510],[19535,19533],[19602,19599],[19613,19609],[19653,19654],[19659,19656],[19682,19680],[19698,19697],[19732,19731],
                        [19814,19812],[20007,20006],[20115,20112],[20345,20344],[20437,20436],[20465,20464],[20632,20631],[20667,20666],[20814,20815],[20827,20826],[20874,20873],[20953,20954],
                        [21048,21047],[21208,21207],[21240,21233],[21647,21646],[21670,21668],[21799,21798],[21971,21970],[21973,21975],[22075,22074],[22260,22259],[22373,22367],[22530,22527],
                        [22547,22549],[22574,22573],[22639,22638],[22808,22802],[22934,22933],[22981,22982],[23056,23053],[23257,23244],[23293,23292],[23300,23299],[23665,23666],[23763,23762],
                        [23782,23781],[23845,23846],[23918,23917],[23923,23922],[24018,24017],[24167,24166],[24172,24171],[24354,24351],[24467,24469],[24763,24762],[24840,24841],[24947,24946],
                        [25118,25117],[25169,25168],[25262,25261],[25341,25340],[25454,25453],[25490,25489],[25823,25822],[26118,26117],[26440,26439],[27774,27769],[27976,27975],[27988,27987],
                        [28015,28014],[28302,28297],[28308,28307],[28442,28441],[28659,28658],[28723,28722],[28986,28985],[28996,28995],[29245,29244],[29258,29254],[29777,29776],[30184,30183],
                        [30476,30475],[30731,30728],[30940,30939],[31103,31102],[31298,31297],[31418,31420],[32029,32026],[32394,32393],[32612,32613],[33056,33055],[33358,33357],[33540,33537],
                        [33696,33695],[33867,33866],[33885,33884],[34130,34129],[34195,34193],[34369,34370],[34435,34429],[34541,34542],[34732,34731],[34771,34770],[34907,34906],[34986,34985],
                        [35059,35058],[35205,35204],[35235,35236],[35390,35389],[35431,35430],[35548,35547],[35834,35835],[36071,36069],[36075,36074],[36120,36119],[36306,36303],[36318,36317],
                        [36509,36508],[36526,36527],[36662,36661],[36682,36681],[36816,36817],[36844,36845],[36953,36952],[36984,36985],[37054,37055],[37088,37087],[37091,37090],[37157,37155],
                        [37192,37191],[37998,37997]]
doublets_gamma_self = sorted(doublets_gamma_self, key=lambda x: min(x))

doublets_n_gamma_self = [[6606,6605],[10030,10028],[10783,10781],[10861,10860],[10864,10862],[10920,10919],[12590,12589],[12923,12922],[13699,13690],[14025,14026],[14172,14169],[14770,14769],
                        [15128,15126],[18030,18029],[18373,18370],[19422,19421],[19698,19697],[20465,20464],[20667,20666],[20874,20873],[20953,20954],[21048,21047],[21208,21207],[23056,23053],
                        [23763,23762],[24167,24166],[24172,24171],[24467,24469],[24840,24841],[25169,25168],[25341,25340],[26118,26117],[30731,30728],[31298,31297],[32394,32393],[32612,32613],
                        [33056,33055],[33358,33357],[34907,34906],[35235,35236],[35548,35547],[35834,35835],[36075,36074],[36120,36119],[36306,36303]]
doublets_n_gamma_self = sorted(doublets_n_gamma_self, key=lambda x: min(x))

doublets_sd_self = [[10030,10028],[10861,10860],[10864,10862],[10920,10919],[12590,12589],[12923,12922],[14025,14026],[14172,14169],[14770,14769],[18030,18029],[18373,18370],[19422,19421],
                    [19698,19697],[20465,20464],[20667,20666],[20874,20873],[20953,20954],[21048,21047],[21208,21207],[24467,24469],[24840,24841],[25169,25168],[25341,25340],[26118,26117],
                    [30731,30728],[31298,31297],[32394,32393],[32612,32613],[33056,33055],[33358,33357],[34907,34906],[35235,35236],[35548,35547],[35834,35835],[36075,36074],[36120,36119]]
doublets_sd_self = sorted(doublets_sd_self, key=lambda x: min(x))


indicator_nu = 'nu'
indicator_sw = 'sw'
indicator_gamma_self = 'gs'
indicator_n_gamma_self = 'ngs'
indicator_sd_self = 'sds'


i_sw = 0
i_gs = 0
i_ngs = 0
i_sd = 0


for doublet in doublets_nu: 
    
    indicator = indicator_nu
    
    if doublet in doublets_sw: indicator += ', ' + indicator_sw
    if doublet in doublets_gamma_self: indicator += ', ' + indicator_gamma_self
    if doublet in doublets_n_gamma_self: indicator += ', ' + indicator_n_gamma_self
    if doublet in doublets_sd_self: indicator += ', ' + indicator_sd_self
    
    indicator += ' {}'.format(doublet)
    print(indicator)      

    df_sceg_save.loc[doublet[0], 'doublet'] = indicator
    df_sceg_save.loc[doublet[1], 'doublet'] = indicator


for doublet in doublets_sw: 
        
    if doublet not in doublets_nu:
            
        indicator = indicator_sw

        if doublet in doublets_gamma_self: indicator += ', ' + indicator_gamma_self
        if doublet in doublets_n_gamma_self: indicator += ', ' + indicator_n_gamma_self
        if doublet in doublets_sd_self: indicator += ', ' + indicator_sd_self
        
        i_sw+=1
        
        indicator += ' {}'.format(doublet)
        print(indicator)
        
        df_sceg_save.loc[doublet[0], 'doublet'] = indicator
        df_sceg_save.loc[doublet[1], 'doublet'] = indicator
        

for doublet in doublets_gamma_self: 
        
    if (doublet not in doublets_nu) and (doublet not in doublets_sw):
        
        indicator = indicator_gamma_self

        if doublet in doublets_n_gamma_self: indicator += ', ' + indicator_n_gamma_self
        if doublet in doublets_sd_self: indicator += ', ' + indicator_sd_self
    
        i_gs+=1
    
        indicator += ' {}'.format(doublet)  
        print(indicator)
        
        df_sceg_save.loc[doublet[0], 'doublet'] = indicator
        df_sceg_save.loc[doublet[1], 'doublet'] = indicator

for doublet in doublets_n_gamma_self: 
        
    if (doublet not in doublets_nu) and (doublet not in doublets_sw) and (doublet not in doublets_gamma_self):
        
        indicator = indicator_n_gamma_self

        if doublet in doublets_sd_self: indicator += ', ' + indicator_sd_self
    
        i_ngs+=1
    
        indicator += ' {}'.format(doublet)  
        print(indicator)
        
        df_sceg_save.loc[doublet[0], 'doublet'] = indicator
        df_sceg_save.loc[doublet[1], 'doublet'] = indicator

for doublet in doublets_sd_self: 
        
    if (doublet not in doublets_nu) and (doublet not in doublets_sw) and (doublet not in doublets_gamma_self) and (doublet not in doublets_n_gamma_self):
        
        indicator = indicator_sd_self

        i_sd+=1
    
        indicator += ' {}'.format(doublet)  
        print(indicator)

        df_sceg_save.loc[doublet[0], 'doublet'] = indicator
        df_sceg_save.loc[doublet[1], 'doublet'] = indicator

                    

df_sceg_save.to_csv(os.path.join(d_sceg,'sceg_pure_output.csv'))



#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%                                               AIR WATER SECTION
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%

#%% water concentration plot



y_h2o_HT = np.array([0.0189748, 0.0191960, 0.0192649, 0.0193174, 0.0193936, 0.0194903, 0.0195316, 0.0194732, 
            0.0193572, 0.0193187, 0.0192066, 0.0192580, 0.0195697, 
            0.0189490, 0.0190159, 0.0189894, 0.0189217, 0.0189221, 
            0.0186053 ,0.0189104 ,0.0187065, 0.0185842, 0.0185690, 
            0.0191551, 0.0195356, 0.0192415, 0.0187509, 0.0188582, 
            0.0193093]) # calculated using 38 features (listed above) using HITRAN 2020

unc_h2o_HT = np.array([0.0004432, 0.0002002, 0.0001270, 0.0001513, 0.0001536, 0.0001736, 0.0001916, 0.0002626, 
              0.0001654, 0.0001314, 0.0001169, 0.0001211, 0.0001679, 
              0.0001678, 0.0001370, 0.0001011, 0.0001104, 0.0001534, 
              0.0002408, 0.0001451, 0.0001385, 0.0001124, 0.0001509, 
              0.0002343, 0.0001797, 0.0001460, 0.0001370, 0.0002026, 
              0.0004038])


y_h2o_sceg = np.array([0.0190805, 0.0193217, 0.0193786, 0.0194300, 0.0195055, 0.0195998, 0.0196325, 0.0195542, 
                0.0194681, 0.0194289, 0.0193158, 0.0193640, 0.0196652, 
                0.0190566, 0.0191279, 0.0190909, 0.0190324, 0.0190234, 
                0.0187246, 0.0190349, 0.0188222, 0.0187054, 0.0186803, 
                0.0193032, 0.0196869, 0.0193919, 0.0188933, 0.0189874, 
                0.0194611]) # calculated using 38 features (listed above) using updated database (~0.0001 lower)

unc_h2o_sceg = np.array([0.0004701, 0.0002979, 0.0003186, 0.0003564, 0.0003744, 0.0004073, 0.0004106, 0.0004500, 
                  0.0001839, 0.0001663, 0.0001938, 0.0002941, 0.0003191, 
                  0.0002280, 0.0001885, 0.0002045, 0.0002451, 0.0002804, 
                  0.0002053, 0.0001453, 0.0001721, 0.0001990, 0.0002662, 
                  0.0003305, 0.0002903, 0.0003035, 0.0002935, 0.0003544, 
                  0.0005751])


y_h2o_update = np.array([0.0194565, 0.0197830, 0.0199033, 0.0199502, 0.0200456, 0.0200617, 0.0200454, 0.0199243, 
                  0.0198340, 0.0199169, 0.0198419, 0.0198157, 0.0200947, 
                  0.0193546, 0.0195809, 0.0196135, 0.0195629, 0.0195001, 
                  0.0189080, 0.0193659, 0.0193067, 0.0192536, 0.0191974, 
                  0.0195506, 0.0199976, 0.0198608, 0.0194793, 0.0195815, 
                  0.0200991]) # calculated floating SD and gamma 

unc_h2o_update = np.array([0.0005702, 0.0003298, 0.0002941, 0.0003280, 0.0002813, 0.0003691, 0.0003581, 0.0005546, 
                  0.0002175, 0.0001416, 0.0001631, 0.0002817, 0.0003259, 
                  0.0002966, 0.0002156, 0.0001906, 0.0002395, 0.0003069, 
                  0.0002830, 0.0001936, 0.0001760, 0.0002290, 0.0003151, 
                  0.0003427, 0.0003291, 0.0003145, 0.0002949, 0.0003812, 
                  0.0005685])

T = np.array([300, 300, 300, 300, 300, 300, 300, 300, 
     500, 500, 500, 500, 500, 
     700, 700, 700, 700, 700, 
     900, 900, 900, 900, 900, 
     1100, 1100, 1100, 1100, 1100, 
     1300])

P = np.array([20, 40, 60, 80, 120, 160, 320, 600, 
     40, 80, 160, 320, 600, 
     40, 80, 160, 320, 600, 
     40, 80, 160, 320, 600, 
     40, 80, 160, 320, 600, 
     600])

colors =            ['k', '#0028ff','#0080af','#117d11','#be961e','#ff0000']
# colors_fade = ['#bfbfbf', '#8093ff','#57d2ff','#5de95d','#ebd182','#ff8080'] # 50
# colors_fade = ['#d3d3d3', '#d3d9ff','#c4efff','#c6f7c6','#f8efd3','#ffd3d3'] # 65
# colors_fade = ['#c8c8c8', '#91a3ff', '#6fd8ff','#74ec74','#eed793','#ff9191'] # 57
# colors_fade = ['#cdcdcd', '#9cabff','#7cdcff','#81ee81','#f0db9d','#ff9c9c'] # 61

colors_fade = ['#cdcdcd', '#9cabff','#7cdcff','#81ee81','#ebd182','#ff9c9c'] # mix


markers = ['v', 'o', '>','s','^','d', '<']
linestyles = ['solid', (5, (10, 3)), 'dashdot', 'dashed','dotted','']

T_plot = list(set(T))
T_plot.sort()


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')

for i, T_i in enumerate(T_plot): 
    
    plot_x = P[T == T_i]
    plot_y_update = y_h2o_update[T == T_i]*100
    plot_unc_updated = unc_h2o_update[T == T_i]*100
    
    plot_y_HT = y_h2o_HT[T == T_i]*100
    plot_unc_HT = unc_h2o_HT[T == T_i]*100
    
    plt.plot(plot_x + T_i/300, plot_y_update, color=colors[i], marker=markers[i], linestyle='solid', 
             label='{} K'.format(T_i), zorder=5)
    plt.errorbar(plot_x + T_i/300, plot_y_update, yerr=plot_unc_updated, ls='none', color=colors[i], zorder=3)
    
    plt.plot(plot_x + T_i/300, plot_y_HT, color=colors_fade[i], marker=markers[i], linestyle='dashed', zorder=1)
    plt.errorbar(plot_x + T_i/300, plot_y_HT, yerr=plot_unc_HT, ls='none', color=colors_fade[i], zorder=1)
    

plt.legend(loc='lower center', ncol=3, edgecolor='k', framealpha=1, labelspacing=0.5, fontsize=12)

# plt.xlim(-.9,24.9)
plt.ylim(1.76,2.075)

ax = plt.gca()
ax.minorticks_on()

plt.xlabel('Pressure (T)', fontsize=12)
plt.ylabel('Optically Measured H$_{2}$O Concentration (%)', fontsize=12)

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 x h2o.svg',bbox_inches='tight')


# %% feature widths - J plot


plot_which_y = 'gamma_self'
label_y = 'Self-Width, γ$_{self}$ [cm$^{-1}$/atm]'

# 
# label_y = 'Self-Width Temperature Exponent, n$_{self}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)] #&(df_sceg['uc_n_self'] > -1)] # floating all width parameters

# df_plot_ht = df_HT2020_HT_align[(df_sceg['uc_gamma_self'] > -1)]
# g_error = df_plot_ht.ierr.str[3]
# g_ref = df_plot_ht.iref.str[6:8]
# g_ref_dict = {}


plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


# plot_x = df_plot[['Kap', 'Kapp']].max(axis=1) + 0.1*(df_plot[['Jp', 'Jpp']].max(axis=1) - df_plot[['Kap', 'Kapp']].max(axis=1))
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]

 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=5000)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
j_HT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
g_HT = [0.50361, 0.47957, 0.45633, 0.43388, 0.41221, 0.39129, 0.37113, 0.3517, 0.333, 0.31501, 
        0.29773, 0.28113, 0.26521, 0.24996, 0.23536, 0.2214, 0.20806, 0.19534, 0.18323, 0.1717, 
        0.16076, 0.15038, 0.14056, 0.13128, 0.12252, 0.11429]

plt.plot(j_HT,g_HT, colors[0], label='HITRAN/HITEMP', linewidth=4)
plt.plot([0, 25], [.484, 0.484-0.018*25], colors[1], label='This Work (0.484-0.018J")',
         linewidth=4, linestyle='dashed')

plt.legend(loc='upper right', ncol=1, edgecolor='k', framealpha=1, labelspacing=0.5)

plt.colorbar(sc, label=label_c, pad=0.01)
plt.show()

ax = plt.gca()
ax.minorticks_on()

plt.xlim(-.9,24.9)
plt.ylim(-0.04,1.19)

plt.xticks(np.arange(0, 25, 2.0))

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width Linda.svg',bbox_inches='tight')

# %% feature temperature dependence of widths - Linda plot


plot_which_y = 'n_self'
label_y = 'Self-Width Temperature Exponent, n$_{γ,self}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)] # floating all width parameters

plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)



# plot_x = df_plot[['Kap', 'Kapp']].max(axis=1) + 0.1*(df_plot[['Jp', 'Jpp']].max(axis=1) - df_plot[['Kap', 'Kapp']].max(axis=1))
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]


 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=3000)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
p = np.polyfit(plot_x, plot_y, 1)
plot_x_sparse = [0, 18]
plot_y_fit = np.poly1d(p)(plot_x_sparse)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

plt.plot([0,18], [0.997, 0.997-0.068*18], colors[1], label='This Work (0.997-0.068J")',
         linewidth=4, linestyle='dashed')

plt.legend(loc='lower left', ncol=2, edgecolor='k', framealpha=1, labelspacing=0)


plt.colorbar(sc, label=label_c, pad=0.01)
plt.show()

plt.xlim(-.9,19.9)
plt.ylim(-0.79,1.3)
plt.xticks(np.arange(0, 19, 2.0))

ax = plt.gca()
ax.minorticks_on()



plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 n width Linda.svg',bbox_inches='tight')





# %% feature widths vs HITRAN

plot_which = 'n_self'
label_y = 'n$_{γ,self}$, This Work [cm$^{-1}$/atm]'
label_x = 'n$_{γ,air}$, HITRAN [cm$^{-1}$/atm]'

label_c = 'Angular Momentum of Ground State, J"'

which = (df_sceg['uc_n_self'] > -1) # &(df_sceg['uc_n_self'] > -1)

df_plot = df_sceg_align[which].sort_values(by=['Jpp'])
df_HT2020_align['Jpp'] = df_sceg_align.Jpp
df_plot_HT = df_HT2020_align[which].sort_values(by=['Jpp'])

# df_plot['gamma_self'] = df_HT2020_align.gamma_self
# df_plot['n_self'] = df_HT2020_align.n_self

# df_plot_ht = df_HT2020_HT_align[(df_sceg['uc_gamma_self'] > -1)]
# g_error = df_plot_ht.ierr.str[3]
# g_ref = df_plot_ht.iref.str[6:8]
# g_ref_dict = {}

# df_plot_HT = df_plot_HT[g_ref == '71']


plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


plot_x = df_plot_HT['n_air']
plot_y = df_plot[plot_which]
plot_c = df_plot_HT.Jpp



 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
if plot_logx: 
    plt.xscale('log')
    
# plt.legend()


cbar = plt.colorbar(sc, label=label_c,  pad=0.01) # pad=-0.95, aspect=10, shrink=0.5), fraction=0.5
# cbar.ax.set_ylabel(label_c, rotation=90, ha='center', va='center')


line_ = [0, 1]

# plt.plot(line_,line_,'k',linewidth=2)

p = np.polyfit(plot_x, plot_y, 1)
plot_y_fit = np.poly1d(p)(line_)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

r2 = r_value**2

plt.plot(line_, plot_y_fit, colors[1], label='This Work  ({}n$_{}${})'.format(str(slope)[:4],'HT',str(intercept)[:5]),
          linewidth=4, linestyle='dashed')

plt.legend(loc='lower right', edgecolor='k', framealpha=1)

plt.show()

# plt.ylim(-.59,1.24)
# plt.xlim(0.05,0.6)

# plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width HT.svg',bbox_inches='tight')


mad = np.mean(np.abs(plot_y-np.poly1d(p)(plot_x)))
rms = np.sqrt(np.sum((plot_y-np.poly1d(p)(plot_x))**2)/ len(plot_y))
r2 = r2_score(plot_y, np.poly1d(p)(plot_x))


print(rms)


# %% feature widths vs temp dependence (matching Paul's style)

colors = ['#d95f02','k']

#-----------------------
plot_which_y = 'n_self'
label_y = 'Self-Width Temperature Exponent, n$_{γ,self}$'

plot_which_x = 'gamma_self'
label_x = 'Self-Width, γ$_{self}$ [cm$^{-1}$/atm]'

label_c = 'Angular Momentum of Ground State, J"'

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)].sort_values(by=['Jpp']).sort_values(by=['Jpp'])

plot_unc_y_bool = True
plot_unc_x_bool = True

plot_labels = False
plot_logx = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)



plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]
plot_c = df_plot.Jpp


 
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
             # label=HT_errors_nu[err])

if plot_unc_x_bool: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, ls='none', color='k', zorder=1)
if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
    
if plot_logx: 
    plt.xscale('log')
    
# plt.legend()


cbar = plt.colorbar(sc, label=label_c,  pad=0.01) # pad=-0.95, aspect=10, shrink=0.5), fraction=0.5
# cbar.ax.set_ylabel(label_c, rotation=90, ha='center', va='center')


p = np.polyfit(plot_x[(plot_x>0.2)], plot_y[(plot_x>0.2)], 1)
plot_x_sparse = [0.2, 0.5]
plot_y_fit = np.poly1d(p)(plot_x_sparse)

slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

plt.plot(plot_x_sparse,[1.5344*.2+0.0502, 1.5344*.5+0.0502], colors[0], label='Schroeder  (0.05+1.53γ)', linewidth=4)
plt.plot(plot_x_sparse, plot_y_fit, colors[1], label='This Work ({}+{}γ)'.format(str(intercept)[:5], str(slope)[:4]),
         linewidth=4, linestyle='dashed')

plt.legend(loc='lower right', edgecolor='k', framealpha=1)


plt.show()

plt.ylim(-.69,1.29)
plt.xlim(0.05,0.57)

plt.xticks(np.arange(0.1, 0.6, 0.1))


ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 width Paul.svg',bbox_inches='tight')



# %% speed dependence of feature widths


colors = ['#d95f02','#514c8e']

#-----------------------
plot_which_y = 'sd_self'
label_y = 'Speed Dependence of the Self-Width, a$_{w}$'

plot_which_x = 'Jpp'
label_x = 'J" + K$_{c}$"/10'

label_c = 'Lower State Energy, E" [cm$^{-1}$]'
plot_which_c = 'elower'

df_plot = df_sceg_align[(df_sceg['uc_gamma_air'] > -1)&(df_sceg['uc_n_air'] > -1)&(df_sceg['uc_sd_self'] > -1)].sort_values(by=['Jpp']) # all width (with SD)

df_plot = df_plot[(df_sceg['uc_gamma_air'] < 0.03)&(df_sceg['uc_n_air'] < 0.03)&(df_sceg['uc_sd_self'] < 0.03)].sort_values(by=['Jpp']) # all width (with SD)


plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False

plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


# plot_x = df_plot[plot_which_x]
plot_x = df_plot[plot_which_x] + df_plot['Kcpp'] / 10
plot_y = df_plot[plot_which_y]
plot_c = df_plot[plot_which_c]

sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=3000)

if plot_unc_y_bool: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, ls='none', color='k', zorder=1)
       
cbar = plt.colorbar(sc, label=label_c,  pad=0.01) # pad=-0.95, aspect=10, shrink=0.5), fraction=0.5


polyfit = np.polyfit(plot_x, plot_y, 0, full=True)
p = polyfit[0]
fit_stats = polyfit[1:]
plot_x_sparse = [0, 19]
plot_y_fit = np.poly1d(p)(plot_x_sparse)


std = np.std(np.poly1d(p)(plot_x) - plot_y)
# r2 = r2_score(plot_y, np.poly1d(p)(plot_x))

plt.plot(plot_x_sparse,[0.125597, 0.125597], colors[0], label='Schroeder Average ({})'.format('0.126'), linewidth=4)
plt.plot(plot_x_sparse, plot_y_fit, color='k', label='This Work Average ({})'.format(str(p[0])[0:5]), linewidth=4, linestyle='dashed')

plt.legend(loc='upper right', edgecolor='k', framealpha=1)

plt.ylim((-0.04, 0.39))
plt.xlim(-.9,19.9)
plt.xticks(np.arange(0, 19, 2.0))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 SD.svg',bbox_inches='tight')

# %% shift 

markers = ['1','2','3','+','x', '.', '.', '.']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']

#-----------------------
plot_which_y = 'delta_air'
label_y = 'Self-Shift, δ$_{air}$ [cm$^{-1}$/atm]'

# plot_which_x = 'elower'
# label_x = 'Lower State Energy, E" [cm$^{-1}$]'

plot_which_x = 'delta_air'
label_x = 'Self-Shift HITRAN'

colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']
colors = ['dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'purple', 'moccasin']


legend_dict = {'101': ['ν$_{1}$+ν$_{3}$', '#1b9e77'],
               '021': ['2ν$_{2}$+ν$_{3}$','#e6ab02'],
               '111': ['ν$_{1}$+ν$_{2}$+ν$_{3}$←ν$_{2}$','#514c8e'],
               '200': ['2ν$_{1}$', '#d95f02'],
               '120': ['ν$_{1}$+2ν$_{2}$', '#4c7c17'], 
               '002': ['2ν$_{3}$', 'firebrick'], 
               '040': ['4ν$_{2}$', 'darkgreen'],
               '031': ['3ν$_{2}$+ν$_{3}$←ν$_{2}$', 'dodgerblue'],
               '000': [' x x x ', 'k']}



# plot_which_x = 'm'
# label_x = 'm'

# plot_which_x = 'nu'
# label_x = 'wavenumber'


plot_unc_y_bool = True

plot_labels = False
plot_logx = False

plot_unc_x_bool = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


i=0


df_plot = df_sceg_align[(df_sceg_align['uc_'+plot_which_y] > -1)] # &(df_sceg['uc_n_delta_self'] > -1)]
df_plot_HT = df_HT2020_align[(df_sceg_align['uc_'+plot_which_y] > -1)]

plot_x = df_plot_HT[plot_which_x]
plot_y = df_plot[plot_which_y]

plt.plot(plot_x, plot_y, '+', color='k', label = 'All (This Work)', linewidth=2)
plt.plot([-0.04, 0.01], [-0.04, 0.01])


r2_score(plot_y, plot_x)

asdfasdfsadf

for vp in df_sceg.vp.unique(): 

    df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg.vp == vp)]
    
    plot_x = df_plot[plot_which_x]
    plot_y = df_plot[plot_which_y]
        
    if len(df_plot) > 0: 
    
        plt.plot(plot_x, plot_y, 'x', color=legend_dict[vp][1], label=legend_dict[vp][0], linewidth=2)
    
        if plot_unc_x_bool: 
            plot_unc_x = df_plot['uc_'+plot_which_x]
            plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color=legend_dict[vp][1], ls='none')
        if plot_unc_y_bool: 
            plot_unc_y = df_plot['uc_'+plot_which_y]
            plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color=legend_dict[vp][1], ls='none')
        
        
    
if plot_logx: 
    plt.xscale('log')
    
plt.legend(loc='lower left', edgecolor='k', framealpha=1, labelspacing=0)


# plt.ylim((-0.65, 0.36))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 shift.svg',bbox_inches='tight')


# %% temperature dependence of the shift (not planning to include this in the final paper)

markers = ['1','2','3','+','x', '.', '.', '.']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']


#-----------------------
plot_which_y = 'n_delta_self'
label_y = 'Self-Shift Temperature Exponent, n$_{δ,self}$'

plot_which_x = 'elower'
label_x = 'Lower State Energy, E" [cm$^{-1}$]'

# -----------------------
# plot_which_y = 'n_delta_self'
# label_y = 'Temp. Dep. of Pressure Shift'

# plot_which_x = 'delta_self'
# label_x = 'Pressure Shift'


legend_dict = {'101': ['ν$_{1}$+ν$_{3}$', '#1b9e77'],
               '021': ['2ν$_{2}$+ν$_{3}$','#e6ab02'],
               '111': ['ν$_{1}$+ν$_{2}$+ν$_{3}$←ν$_{2}$','#514c8e'],
               '200': ['2ν$_{1}$', '#d95f02'],
               '120': ['ν$_{1}$+2ν$_{2}$', '#4c7c17'], 
               '002': ['2ν$_{3}$', 'firebrick'], 
               '040': ['4ν$_{2}$', 'darkgreen'],
               '031': ['3ν$_{2}$+ν$_{3}$←ν$_{2}$', 'dodgerblue']}


plot_unc_y_bool = True

plot_labels = False
plot_logx = False

plot_unc_x_bool = False


plt.figure(figsize=(6.5, 3.6)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x, fontsize=12)
plt.ylabel(label_y, fontsize=12)


i=0


df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)]
plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]

# plt.plot(plot_x, plot_y, '+', color='k', label = 'All (This Work)', linewidth=2)

for vp in df_sceg.vp.unique(): 

    df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg.vp == vp)]
    
    plot_x = df_plot[plot_which_x]
    plot_y = df_plot[plot_which_y]
        
    if len(df_plot) > 0: 
    
        plt.plot(plot_x, plot_y, 'x', color=legend_dict[vp][1], label=legend_dict[vp][0], linewidth=2)
    
        if plot_unc_x_bool: 
            plot_unc_x = df_plot['uc_'+plot_which_x]
            plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color=legend_dict[vp][1], ls='none')
        if plot_unc_y_bool: 
            plot_unc_y = df_plot['uc_'+plot_which_y]
            plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color=legend_dict[vp][1], ls='none')
        
        
    
if plot_logx: 
    plt.xscale('log')
    
plt.legend(loc='lower right', edgecolor='k', framealpha=1, labelspacing=0)

plt.ylim((-0.4, 5.5))

ax = plt.gca()
ax.minorticks_on()

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 n shift.svg',bbox_inches='tight')


# %% table of results (number of fits, vibrational bands, etc.)


props_which = ['uc_nu','uc_sw','uc_gamma_self','uc_n_self','uc_sd_self','uc_delta_self','uc_n_delta_self', 'uc_elower']

count = np.zeros((50,len(props_which)+3))
count_name = []

j = 0

for local_iso_id_iter in df_sceg_align.local_iso_id.unique(): 
    
    df_sceg_iso = df_sceg_align[df_sceg_align.local_iso_id == local_iso_id_iter]
       
    for vp_iter in df_sceg_iso.vp.unique(): 
        
        df_sceg_vp = df_sceg_iso[df_sceg_iso.vp == vp_iter]
        
        for i, prop in enumerate(props_which): 
            
            count[j,i] = len(df_sceg_vp[df_sceg_vp[prop] > 0])
            
        if count[j,:].any(): 
            
            # iso, vp, vpp, J min, J max, 
            
            count_name.append(vp_iter + '-' + df_sceg_vp.vpp.mode()[0])
            
            
            if count_name[j] == '-': count_name[j] = '-2-2-2?'
            
            print(count_name[j])
            
            count[j,-1] = max(df_sceg_vp.Jpp)
            count[j,-2] = min(df_sceg_vp.Jpp)
            
            count[j,-3] = local_iso_id_iter
        
            print(count[j,:])
                        
            j+=1
                
                
            
            
df_sceg_new = df_sceg[df_sceg.index > 100000]

for i, prop in enumerate(props_which): 

    count[-1,i] = len(df_sceg_new[df_sceg_new[prop] > 0])
            


#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%                                               compiling results
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%


# %% table of results (number of fits, vibrational bands, etc.)


props_which = ['uc_nu','uc_sw','uc_gamma_self','uc_n_self','uc_sd_self','uc_delta_self','uc_n_delta_self', 'uc_elower']

count = np.zeros((50,len(props_which)+3))
count_name = []

j = 0

for local_iso_id_iter in df_sceg_align.local_iso_id.unique(): 
    
    df_sceg_iso = df_sceg_align[df_sceg_align.local_iso_id == local_iso_id_iter]
       
    for vp_iter in df_sceg_iso.vp.unique(): 
        
        df_sceg_vp = df_sceg_iso[df_sceg_iso.vp == vp_iter]
        
        for i, prop in enumerate(props_which): 
            
            count[j,i] = len(df_sceg_vp[df_sceg_vp[prop] > 0])
            
        if count[j,:].any(): 
            
            # iso, vp, vpp, J min, J max, 
            
            count_name.append(vp_iter + '-' + df_sceg_vp.vpp.mode()[0])
            
            
            if count_name[j] == '-': count_name[j] = '-2-2-2?'
            
            print(count_name[j])
            
            count[j,-1] = max(df_sceg_vp.Jpp)
            count[j,-2] = min(df_sceg_vp.Jpp)
            
            count[j,-3] = local_iso_id_iter
        
            print(count[j,:])
                        
            j+=1
                
                
            
            
df_sceg_new = df_sceg[df_sceg.index > 100000]

for i, prop in enumerate(props_which): 

    count[-1,i] = len(df_sceg_new[df_sceg_new[prop] > 0])
            

# %% table of uncertainty values updated


props_which_part = ['nu','sw']
unc_where =        [   0,   1]
ref_where =        [   0,   1]

# for i, prop in enumerate(props_which_part): 

which = [df_sceg_align['uc_'+prop] > 0]

unc_code = df_plot_ht.ierr.str[unc_where]


df_plot = df_sceg_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
df_plot_og = df_HT2020_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])

df_plot_ht = df_HT2020_HT_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
nu_error = df_plot_ht.ierr.str[0]
    



























