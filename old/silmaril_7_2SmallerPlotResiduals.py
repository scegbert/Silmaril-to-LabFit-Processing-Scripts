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
d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
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

wide = [6615-50, 7650+25]
narrow = [7094.52, 7096.06]
linewidth = 1

offset1 = 0.05
offset0 = 0.05*25

colors = ['#0028ff','#0080af','#117d11','#be961e', '#ff4000','#ff0000',     '#e6ab02', '#fee9ac']
# colors = ['#FFD700','#FF7F50','#EE82EE','#4169E1', '#00BFFF','#00FFFF',     '#e6ab02']


which_files_partial = ['300 K 16 T', '500 K 16 T', '700 K 16 T', '900 K 16 T', '1100 K 16 T', '1300 K 16 T']

fig = plt.figure(figsize=(12, 4))
gs = GridSpec(2, 2, width_ratios=[2, 1]) #, hspace=0.015, wspace=0.005) # rows, columns

for which_file in which_files_partial: 
    
    i = which_files.index(which_file)
    
    ax00 = fig.add_subplot(gs[0,0]) # First row, first column
    ax00.axvline(narrow[0]-offset0, linewidth=1, color=colors[-2])
    ax00.axvline(narrow[1]+offset0, linewidth=1, color=colors[-2])
    ax00.plot(wvn[i],trans[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
    ax00.legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=9)
    
    ax01 = fig.add_subplot(gs[0,1]) # First row, second column
    ax01.plot(wvn[i],trans[i], color=colors[i], 
              linewidth=linewidth)
    
    ax10 = fig.add_subplot(gs[1,0]) # Second row, first column
    ax10.axvline(narrow[0]-offset0, linewidth=1, color=colors[-2])
    ax10.axvline(narrow[1]+offset0, linewidth=1, color=colors[-2])
    ax10.plot(wvn[i],res_updated[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
    # ax10.legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=9)
    
    ax11 = fig.add_subplot(gs[1,1]) # Second row, second column
    ax11.plot(wvn[i],res_updated[i], color=colors[i], 
              linewidth=linewidth)


#%% arrows pointing to inset

# ax00.arrow(narrow[1], 0.5, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
# ax10.arrow(narrow[1], 0.27, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')

#%% set axis
ax00.set_xlim(wide)
ax01.set_xlim(narrow)

ax01.set_ylim([0.861, 0.886])
ax11.set_ylim([-0.1,0.1])
ax00.set_ylim([0,1.1])
ax10.set_ylim([-0.1,0.1])


#%%  remove x label on upper plots (mostly covered)
plt.setp(ax00.get_xticklabels(), visible=False) 
plt.setp(ax01.get_xticklabels(), visible=False)


#%% move zoomed y labels to the right
ax01.yaxis.set_label_position("right")
ax01.yaxis.tick_right()
ax11.yaxis.set_label_position("right")
ax11.yaxis.tick_right()


# %% add ticks and minor ticks all over
ax00.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax00.xaxis.set_minor_locator(AutoMinorLocator(10))
ax00.yaxis.set_minor_locator(AutoMinorLocator(10))

ax10.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax10.xaxis.set_minor_locator(AutoMinorLocator(10))
ax10.yaxis.set_minor_locator(AutoMinorLocator(10))

ax01.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax01.xaxis.set_minor_locator(AutoMinorLocator(5))
ax01.yaxis.set_minor_locator(AutoMinorLocator(5))

ax11.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax11.xaxis.set_minor_locator(AutoMinorLocator(5))
ax11.yaxis.set_minor_locator(AutoMinorLocator(5))


#%% shading to highlight zoomed region
alpha = 1

ax00.axvspan(narrow[0]-offset0, narrow[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax10.axvspan(narrow[0]-offset0, narrow[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax01.axvspan(narrow[0]+offset1, narrow[1]-offset1*1.2, alpha=alpha, color=colors[-1], zorder=0)
ax11.axvspan(narrow[0]+offset1, narrow[1]-offset1*1.2, alpha=alpha, color=colors[-1], zorder=0)


#%% labels
ax11.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax10.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

ax00.set_ylabel('Intensity (arb.)')
ax10.set_ylabel('Meas-Model')


#%%

h0 = 0.02
h1 = 0.05
v0 = 0.9
v1 = 0.9

ax00.text(h0, v1, "A", fontweight="bold", fontsize=12, transform=ax00.transAxes)
ax01.text(h1, v1, "B", fontweight="bold", fontsize=12, transform=ax01.transAxes)
ax10.text(h0, v1, "C", fontweight="bold", fontsize=12, transform=ax10.transAxes)
ax11.text(h1, v1, "D", fontweight="bold", fontsize=12, transform=ax11.transAxes)


#%% save it

# plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-1big.svg',bbox_inches='tight')

