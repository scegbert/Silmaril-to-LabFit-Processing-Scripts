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
plt.plot(j_HT, 0.485*np.exp(-0.0633*np.array(j_HT)) + 0.04, colors[1], label='This Work (0.485 e$^{-0.0633J"}$ + 0.04)',
         linewidth=4, linestyle='dashed')



plt.legend(loc='upper right', ncol=1, edgecolor='k', framealpha=1, labelspacing=0.5)

cbar = plt.colorbar(sc, label=label_c, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(label=label_c, size=12)
plt.show()

ax = plt.gca()
ax.minorticks_on()

plt.xlim(-.9,24.9)
plt.ylim(-0.04,1.19)

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
            


#%% facts about uncatalogued features







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


y_h2o_update = np.array([0.0190805, 0.0193217, 0.0193786, 0.0194300, 0.0195055, 0.0195998, 0.0196325, 0.0195542, 
                0.0194681, 0.0194289, 0.0193158, 0.0193640, 0.0196652, 
                0.0190566, 0.0191279, 0.0190909, 0.0190324, 0.0190234, 
                0.0187246, 0.0190349, 0.0188222, 0.0187054, 0.0186803, 
                0.0193032, 0.0196869, 0.0193919, 0.0188933, 0.0189874, 
                0.0194611]) # calculated using 38 features (listed above) using updated database (~0.0001 lower)

unc_h2o_update = np.array([0.0004701, 0.0002979, 0.0003186, 0.0003564, 0.0003744, 0.0004073, 0.0004106, 0.0004500, 
                  0.0001839, 0.0001663, 0.0001938, 0.0002941, 0.0003191, 
                  0.0002280, 0.0001885, 0.0002045, 0.0002451, 0.0002804, 
                  0.0002053, 0.0001453, 0.0001721, 0.0001990, 0.0002662, 
                  0.0003305, 0.0002903, 0.0003035, 0.0002935, 0.0003544, 
                  0.0005751])

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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')

for i, T_i in enumerate(T_plot): 
    
    plot_x = P[T == T_i]
    plot_y_update = y_h2o_update[T == T_i]*100
    plot_unc_updated = unc_h2o_update[T == T_i]*100
    
    plot_y_HT = y_h2o_HT[T == T_i]*100
    plot_unc_HT = unc_h2o_HT[T == T_i]*100
    
    plt.plot(plot_x + T_i/300, plot_y_update, color=colors[i], marker=markers[i], linestyle='solid', 
             label='{} K'.format(T_i), zorder=5)
    plt.errorbar(plot_x + T_i/300, plot_y_update, yerr=plot_unc_updated, ls='none', color=colors[i], zorder=3)
    
    plt.plot(plot_x + T_i/300, plot_y_HT, color=colors_fade[i], marker=markers[i], linestyle='solid', zorder=1)
    plt.errorbar(plot_x + T_i/300, plot_y_HT, yerr=plot_unc_HT, ls='none', color=colors_fade[i], zorder=1)
    

plt.legend(loc='lower center', ncol=3, edgecolor='k', framealpha=1, labelspacing=0.5)

# plt.xlim(-.9,24.9)
plt.ylim(1.78,2.02)

ax = plt.gca()
ax.minorticks_on()

plt.xlabel('Pressure (T)')
plt.ylabel('Optically Measured Water Concentration (%)')

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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)



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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)



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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)


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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)



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

df_plot = df_sceg_align[(df_sceg['uc_gamma_self'] > -1)&(df_sceg['uc_n_self'] > -1)&(df_sceg['uc_sd_self'] > -1)].sort_values(by=['Jpp']) # all width (with SD)

plot_unc_y_bool = True
plot_unc_x_bool = False

plot_labels = False
plot_logx = False

plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)


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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)


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
    
plt.legend(loc='lower left', edgecolor='k', framealpha=1, labelspacing=0)


plt.ylim((-0.65, 0.36))

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


plt.figure(figsize=(7.2, 4)) #, dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)


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
    



























