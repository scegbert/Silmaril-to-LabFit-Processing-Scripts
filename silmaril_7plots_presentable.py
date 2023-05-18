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

HT_errors_nu = {'0': '0 (>1E0)', 
                '1': '1 (<1E0)', 
                '2': '2 (<1E-1)', 
                '3': '3 (<1E-2)', 
                '4': '4 (<1E-3)', 
                '5': '5 (<1E-4)', 
                '6': '6 (<1E-5)', 
                '7': '7 (<1E-6)', 
                '8': '8 (<1E-7)', 
                '9': '9 (<1E-8)'}


# %% other stuff to put at the top here


markers = ['1','2','3', '+', 'x']
linestyles = [(5, (10, 3)), 'dashed', 'dotted', 'dashdot', 'solid']

colors = ['dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'purple', 'moccasin']
colors_grad = ['firebrick','orangered','goldenrod','forestgreen','teal','royalblue','mediumpurple', 'darkmagenta']



# %% read in results, re-write quantum assignments in a way that is useful


f = open(os.path.join(d_sceg,'df_sceg.pckl'), 'rb')
[df_sceg, df_HT2020, df_HT2020_HT, df_paul] = pickle.load(f)
f.close()

# f = open(os.path.join(d_sceg,'spectra_air.pckl'), 'rb')
# [T_air, P_air, wvn_air, trans_air, res_air, res_og_air] = pickle.load(f)
# f.close()

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure, res_HT_pure] = pickle.load(f)
f.close()

df_sceg_align, df_HT2020_align = df_sceg.align(df_HT2020, join='inner', axis=0)
df_sceg_align2, df_HT2020_HT_align = df_sceg_align.align(df_HT2020_HT, join='inner', axis=0)

if not df_sceg_align.equals(df_sceg_align2): throw = errorplease # these should be the same dataframe if everything lines up

# df_sceg['sw1300'] = lab.strength_T(1300, df_sceg.elower, df_sceg.nu) * df_sceg.sw
# df_HT2020['sw1300'] = lab.strength_T(1300, df_HT2020.elower, df_HT2020.nu) * df_HT2020.sw

please = stophere



# %% percent change SW with E lower

buffer = 1.3

df_plot = df_sceg_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])
df_plot_og = df_HT2016_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])

df_plot_ht = df_HT2020_HT_align[df_sceg_align.uc_sw > 0].sort_values(by=['elower'])
sw_error = df_plot_ht.ierr.str[1]
                                                                   

label_x = 'Line Strength, S$_{296}$ (updated) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Relative Change in Line Strength, S$_{296}$ \n (updated - HITRAN) / HITRAN' # ' \n [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_y'] = (df_plot.sw - df_plot_og.sw) / df_plot_og.sw 
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_sw
label_c = 'Lower State Energy [cm$^{-1}$]'
plot_c = df_plot.elower

plt.figure(figsize=(10, 5), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.errorbar(plot_x,plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)


limited = ['6']



for i, ierr in enumerate(np.sort(sw_error.unique())): 
    
    label = HT_errors[ierr]
    if ierr == '7': label=''
    
    sc = plt.scatter(plot_x[sw_error == ierr], plot_y[sw_error == ierr], marker=markers[i], 
                     c=plot_c[sw_error == ierr], cmap='viridis', zorder=2, 
                     label=label)
    df_plot.sort_values(by=['sw'], inplace=True)
    
    
    if ierr != '3': 
        within_HT = plot_y[sw_error == ierr].abs()
        within_HT = len(within_HT[within_HT < float(HT_errors[ierr].split('-')[-1].split('%')[0])])
    else: within_HT = 'N/A'
    
    print(' {} total, {} within uncertainty'.format(len(plot_x[sw_error == ierr]), within_HT))

plt.legend()

ax = plt.gca()
legend = ax.get_legend()
legend_dict = {handle.get_label(): handle for handle in legend.legendHandles}


for i, ierr in enumerate(np.sort(sw_error.unique())): 
    
    if ierr not in ['7', '6', '5', '3']: 
        
        plt.hlines(float(HT_errors[ierr].split('-')[-1].split('%')[0])/100,min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i])
        plt.hlines(-float(HT_errors[ierr].split('-')[-1].split('%')[0])/100,min(plot_x), max(plot_x),
                    linestyles=linestyles[i], color=colors[i])
        
        
    if ierr not in ['7']: 
        legend_dict[HT_errors[ierr]].set_color(colors[i])
        df_plot.sort_values(by=['sw'], inplace=True)




plt.xlim(min(plot_x)/buffer, max(plot_x)*buffer)
plt.xscale('log')

plt.colorbar(sc, label=label_c)
plt.show()



# plot inset 

# ax_ins = inset_axes(ax, width='40%', height='50%', loc='upper center', bbox_to_anchor=(0.05,0,1,1), bbox_transform=ax.transAxes)
ax_ins = inset_axes(ax, width='50%', height='35%', loc='center right', bbox_to_anchor=(0,-0.05,1,1), bbox_transform=ax.transAxes)

for i, ierr in enumerate(np.sort(sw_error.unique())): 
    
    if ierr not in ['7']: 
    
        ax_ins.scatter(plot_x[sw_error == ierr], plot_y[sw_error == ierr], marker=markers[i], 
                         c=plot_c[sw_error == ierr], cmap='viridis', zorder=2, 
                         label=HT_errors[ierr])
        df_plot.sort_values(by=['sw'], inplace=True)
        
        legend_dict[HT_errors[ierr]].set_color(colors[i])
        
        if ierr not in ['3', '7']: 
            plt.hlines(float(HT_errors[ierr].split('-')[-1].split('%')[0])/100,min(plot_x), max(plot_x),
                        linestyles=linestyles[i], color=colors[i])
            plt.hlines(-float(HT_errors[ierr].split('-')[-1].split('%')[0])/100,min(plot_x), max(plot_x),
                        linestyles=linestyles[i], color=colors[i])
        

patch, pp1,pp2 = mark_inset(ax, ax_ins, loc1=1, loc2=2, fc='none', ec='k', zorder=0)
pp1.loc1 = 3
pp2.loc1 = 4


# updates for entire plot


plt.xlim(2e-26, 2.5e-20)
plt.ylim(-0.12, 0.12)

plt.xscale('log')


plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7 SW.svg',bbox_inches='tight')



# %% change in line center with E lower

buffer = 1.3

df_plot = df_sceg_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
df_plot_og = df_HT2020_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])

df_plot_ht = df_HT2020_HT_align[df_sceg_align.uc_nu > 0].sort_values(by=['elower'])
sw_error = df_plot_ht.ierr.str[1]


label_x = 'Line Strength, S$_{296}$ (updated) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Difference in Line Position, $\Delta\\nu$ \n (updated - HITRAN) [cm$^{-1}$]'
df_plot['plot_y'] = df_plot.nu - df_plot_og.nu
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_nu
label_c = 'Lower State Energy [cm$^{-1}$]'
plot_c = df_plot.elower

limited = ['6']

# for vp in df_plot.vp.unique():     

# if len(df_plot[df_plot.vp == vp]) > 140: 

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

# plt.title(vp)

for i_err, err in enumerate(np.sort(sw_error.unique())): 
    
    which = (sw_error == err) #&(df_plot.vp == vp)
    
    plt.errorbar(plot_x[which],plot_y[which], yerr=plot_y_unc[which], color='k', ls='none', zorder=1)
    
    sc = plt.scatter(plot_x[which], plot_y[which], marker=markers[i], 
                     c=plot_c[which], cmap='viridis', zorder=2, 
                     label=HT_errors_nu[err])
    df_plot.sort_values(by=['sw'], inplace=True)
    
    within_HT = plot_y[which].abs()
    within_HT = len(within_HT[within_HT < float(HT_errors_nu[err].split(')')[0].split('<')[-1])])

    
    print('{} -  {} total, {} within uncertainty'.format(err,len(plot_x[which]), within_HT))
    
    delta_avg = np.mean(plot_y[which])
    delta_avg_abs = np.mean(abs(plot_y[which]))
    
    print('       {}       {}'.format(delta_avg, delta_avg_abs))

plt.legend(loc='upper right')
ax = plt.gca()
legend = ax.get_legend()
legend_dict = {handle.get_label(): handle for handle in legend.legendHandles}

for i_err, err in enumerate(np.sort(sw_error.unique())): 
    
    # legend_dict[HT_errors_nu[err]].set_color(colors[i])
    
    plt.hlines(float(HT_errors_nu[err].split(')')[0].split('<')[-1]),min(plot_x), max(plot_x),
               linestyles=linestyles[i_err], color=colors[i_err])
    plt.hlines(-float(HT_errors_nu[err].split(')')[0].split('<')[-1]),min(plot_x), max(plot_x),
               linestyles=linestyles[i_err], color=colors[i_err])

plt.xlim(min(plot_x)/buffer, max(plot_x)*buffer)
# plt.ylim(-0.1899, 0.1599)
# plt.ylim(-5e-4, 5e-4)
# plt.ylim(-0.035, 0.035)

plt.xscale('log')

plt.colorbar(sc, label=label_c)

plt.show()



# %% feature widths vs temp dependence (matching Paul's style)


#-----------------------
plot_which_y = 'n_self'
label_y = 'Temperature Exponent\n(n$_{self}$)'

plot_which_x = 'gamma_self'
label_x = 'Self-Broadening Coefficient ($\mathregular{cm^{-1}/atm}$))'

label_c = 'J"' # 'Angular Momentum of Ground State (J")'

df_plot = df_sceg_align[(df_sceg['uc_'+plot_which_x] > -1)&(df_sceg['uc_'+plot_which_y] > -1)&
                        (df_sceg['uc_'+plot_which_x] < 0.05)&(df_sceg['uc_'+plot_which_y] < 0.05)].sort_values(by=['Jpp'])

plot_unc_y_bool = True
plot_unc_x_bool = True

plot_labels = False
plot_logx = False


plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)



plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]
plot_c = df_plot.Jpp


 
sc = plt.scatter(plot_x, plot_y, marker=markers[0], c=plot_c, cmap='gist_rainbow', zorder=2, linewidth=2)
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


cbar = plt.colorbar(sc, label=label_c, orientation='vertical') # pad=-0.95, aspect=10, shrink=0.5), fraction=0.5
cbar.ax.set_ylabel(label_c, rotation=90, ha='center', va='center')




p = np.polyfit(plot_x, plot_y, 1)
plot_y_fit = np.poly1d(p)(plot_x)


import scipy.stats as ss
slope, intercept, r_value, p_value, std_err = ss.linregress(plot_x, plot_y)

# plt.plot([0.2, 0.5],[1.5344*.2+0.0502, 1.5344*.5+0.0502], 'r', label='Schroeder Fit Line', linewidth=2)
# plt.plot(plot_x[(plot_x>0.07)&(plot_x<0.55)], plot_y_fit[(plot_x>0.07)&(plot_x<0.55)], 'k', label='Updated Fit Line', linewidth=2)

# plt.legend(loc='upper left')


plt.show()

plt.ylim(-.59,1.24)
plt.xlim(0.05,0.6)

# %% feature widths


#-----------------------
plot_which_y = 'gamma_self'
label_y = 'Self-Width (γ$_{self}$)'

plot_which_x = 'Jpp'
label_x = 'Temperature Dependence of the Self-Width (n$_{self}$)'

label_c = 'Angular Momentum of Ground State (J")'

df_plot = df_sceg_align[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg['uc_'+plot_which_y] < 0.05)].sort_values(by=['Jpp'])

plot_unc_y_bool = True
plot_unc_x_bool = True

plot_labels = False
plot_logx = False


plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)



plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y]
plot_c = df_plot.Jpp


 
sc = plt.scatter(plot_x, plot_y, marker=markers[i], c=plot_c, cmap='viridis', zorder=2)
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

plt.colorbar(sc, label=label_c)
plt.show()

# plt.xlim(-.59,1.24)
# plt.ylim(0.05,0.6)



# %% temp dependence of shift (settings from previous plots)

#-----------------------
plot_which_y = 'n_delta_self'
label_y = 'Temp. Dep. of Pressure Shift'

plot_which_x = 'elower'
label_x = 'Lower State Energy'


#-----------------------
# plot_which_y = 'n_delta_self'
# label_y = 'Temp. Dep. of Pressure Shift'

# plot_which_x = 'delta_self'
# label_x = 'Pressure Shift'


#-----------------------
# plot_which_y = 'delta_self'
# label_y = 'Pressure Shift'

# plot_which_x = 'm'
# label_x = 'm'


plot_unc_y_bool = True

plot_labels = False
plot_logx = False

plot_unc_x_bool = False


plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)


i=0
for vp in df_sceg.vp.unique(): 

    df_plot = df_sceg[(df_sceg['uc_'+plot_which_y] > -1)&(df_sceg.vp == vp)]
    
    plot_x = df_plot[plot_which_x]
    plot_y = df_plot[plot_which_y]
     
    if len(df_plot) > 5: 
    
        plt.plot(plot_x, plot_y, 'x', color=colors[i], label = vp)
    
        if plot_unc_x_bool: 
            plot_unc_x = df_plot['uc_'+plot_which_x]
            plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color=colors[i], ls='none')
        if plot_unc_y_bool: 
            plot_unc_y = df_plot['uc_'+plot_which_y]
            plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color=colors[i], ls='none')
        
        if plot_labels:
            for j in df_plot.index:
                j = int(j)
                plt.annotate(str(j),(plot_x[j], plot_y[j]))
    
        if plot_which_y == 'delta_self': 
            
            df_plot2 = df_plot[(df_plot['uc_'+plot_which_y] > -1)&(df_plot['uc_n_delta_self'] > -1)]
            
            plot_x = df_plot2[plot_which_x]
            plot_y = df_plot2[plot_which_y]
            
            plt.plot(plot_x, plot_y, 'o', color=colors[i+1], markersize=3, label = 'with temp. dep.')
        
        i+=1
        
    
if plot_logx: 
    plt.xscale('log')
    
plt.legend()




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
            
            
            if count_name[j] == '-': asdfasdfasdf
            
            print(count_name[j])
            
            count[j,-1] = max(df_sceg_vp.Jpp)
            count[j,-2] = min(df_sceg_vp.Jpp)
            
            count[j,-3] = local_iso_id_iter
        
            print(count[j,:])
                        
            j+=1
                
                
            
            
df_sceg_new = df_sceg[df_sceg.index > 100000]

for i, prop in enumerate(props_which): 

    count[-1,i] = len(df_sceg_new[df_sceg_new[prop] > 0])
            










