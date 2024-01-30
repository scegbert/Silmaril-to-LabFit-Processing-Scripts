r'''

silmaril 7-1 big plots

makes the big plot of spectra (vacuum, raw, transmission, with inset)


r'''

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

import pldspectrapy as pld
import td_support as td

import numpy as np

#%% setup things

wvn2_processing = [6500, 7800] # range used when processing the data
wvn2_data = [6615, 7650] # where there is actually useful data that we would want to include


#%% load in transmission data (model from labfit results)

# load in labfit stuff (transmission, wvn, residuals before and after, conditions)
# d_sceg = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'
d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
# [T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_HT_pure, res_sd0] = pickle.load(f)
[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_HT_pure] = pickle.load(f)
f.close()


[T_all, P_all] = np.asarray([T_pure, P_pure])


#%% plot and verify this is what you want


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


#%%

adjust = False

# wide = [6770.25, 6773.4] # <---- overview plot range
# y_lim_top = [0.902,1.003]
# y_lim_bottom = [-0.011,0.011]
# fig = plt.figure(figsize=(12, 4))

# wide = [6771.3, 6772.1] # <---- overview plot mini
# y_lim_top = [0.902,1.003]
# y_lim_bottom = [-0.019,0.019]
# fig = plt.figure(figsize=(4, 4))

# wide = [6938.14, 6939.05] # line strength changes
# y_lim_top = [0.972,1.005]
# y_lim_bottom = [-0.009,0.004]
# fig = plt.figure(figsize=(7.2, 4))

wide = [7377.29, 7378.36] # line strength changes #2 ************************* in the paper
wide = [7377.07, 7378.36] # line strength changes #2
y_lim_top = [0.972,1.007]
y_lim_bottom = [-0.008,0.003]
fig = plt.figure(figsize=(6.5, 4)) #, dpi=200, facecolor='w', edgecolor='k')
adjust = True # adjust residual to compensate for the large feature next to the plot (that's not what we're trying to highlight)

# wide = [6718.01, 6719.5] # new feature
# y_lim_top = [0.95,1.003]
# y_lim_bottom = [-0.019,0.004]
# fig = plt.figure(figsize=(7.2, 4))


linewidth = 1

offset1 = 0.05
offset0 = 0.05*25

colors = ['k', '#0028ff','#0080af','#117d11','#be961e','#ff0000',     '#e6ab02', '#fee9ac']
# colors = ['#FFD700','#FF7F50','#EE82EE','#4169E1', '#00BFFF','#00FFFF',     '#e6ab02']


num_files = 6


which_files_partial = which_files[:num_files]

gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.015, wspace=0.005) # rows, columns

for i, which_file in enumerate(which_files_partial): 
    
    if i == len(which_files_partial)-1: linewidth = 1.6
    
    ax00 = fig.add_subplot(gs[0,0]) # First row, first column
    ax00.plot(wvn[i],trans[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
    ax00.legend(loc = 'lower left', framealpha=1, edgecolor='black', fontsize=10)
    
    
    if adjust: 
        if i == 0: res_HT_og = res_HT.copy()
        if i == 1:  res_HT[i] += 0.002
        if i == 2:  res_HT[i] += 0.003
        
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00) # Second row, first column
    ax10.plot(wvn[i],res_HT[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
    
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00) # Second row, first column
    ax20.plot(wvn[i],res_updated[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
    
#%% arrows pointing to inset

# ax00.arrow(narrow[1], 0.5, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
# ax10.arrow(narrow[1], 0.27, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')

#%% set axis
ax00.set_xlim(wide)

ax00.set_ylim(y_lim_top)
ax10.set_ylim(y_lim_bottom)
ax20.set_ylim(y_lim_bottom)



#%%  remove x label on upper plots (mostly covered)
plt.setp(ax00.get_xticklabels(), visible=False) 
plt.setp(ax10.get_xticklabels(), visible=False) 



# %% add ticks and minor ticks all over
ax00.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax00.xaxis.set_minor_locator(AutoMinorLocator(10))
ax00.yaxis.set_minor_locator(AutoMinorLocator(10))

ax10.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax10.xaxis.set_minor_locator(AutoMinorLocator(10))
ax10.yaxis.set_minor_locator(AutoMinorLocator(10))

ax20.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax20.xaxis.set_minor_locator(AutoMinorLocator(10))
ax20.yaxis.set_minor_locator(AutoMinorLocator(10))


#%% labels
ax20.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

ax00.set_ylabel('Measured\nTransmission')
ax10.set_ylabel('Meas-\nHITRAN')
ax20.set_ylabel('Meas-\nTW')



#%%

if adjust:  
    
    v = 0.875
    # ax00.text(0.17, v, '9%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    # ax00.text(0.295, v, '0%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    # ax00.text(0.625, v, '1247%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    # ax00.text(0.76, v, '7%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    # ax00.text(0.895, v, '5%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    
    ax00.text(0.02, v, 'ΔS$_{296}$ vs HT20 =', fontsize=12, transform=ax00.transAxes)
    
    ax00.text(0.305, v, '9%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    ax00.text(0.41, v, '0%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    ax00.text(0.66, v, '1247%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    ax00.text(0.805, v, '7%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    ax00.text(0.91, v, '5%', fontweight="bold", fontsize=12, transform=ax00.transAxes)
    
    
    


#%% save it



plt.savefig(r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\plots\7-2 changes.svg', bbox_inches='tight')

