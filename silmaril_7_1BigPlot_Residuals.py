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

d_type = 'air' # 'air'

# load in labfit stuff (transmission, wvn, residuals before and after, conditions)
d_sceg = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

if d_type == 'pure': 
    f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
elif d_type == 'air': 
    f = open(os.path.join(d_sceg,'spectra_air.pckl'), 'rb')
[T_pure, P_pure, wvn_pure, trans_pure, res_SDVP_raw, res_HT_raw, res_VP_raw] = pickle.load(f)
f.close()


[T_all, P_all] = np.asarray([T_pure, P_pure])

if d_type == 'air': P_all = np.round(P_all/10,0)*10 # for air-water to round pressures to nearest 10's


#%% plot and verify this is what you want

if d_type == 'pure': 

    # which_files = ['300 K _5 T', '300 K 1 T',  '300 K 1_5 T','300 K 2 T',  '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
    #                '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
    #                '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
    #                '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
    #                '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T', 
    #                '1300 K 16 T']

    which_files = ['1300 K 16 T']
    y_h2o = ''
    post_label = ''
    
elif d_type == 'air': 
    
    # which_files = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
    #                 '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
    #                 '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
    #                 '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
    #                 '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T', '1300 K 600 T']
    
    which_files = ['1300 K 600 T']
    y_h2o = '2%'
    post_label = ' in Air'


res_HT = []
res_og = []
res_SDVP = []
res_VP = []
trans = []
wvn = []

RMS_HT = [None] * len(which_files)
RMS_SDVP = [None] * len(which_files)
RMS_VP = [None] * len(which_files)
Prms = [None] * len(which_files)

for i, which_file in enumerate(which_files): 

    T_plot = float(which_file.split()[0])
    P_plot = float(which_file.split()[2].replace('_','.'))
    
    i_plot = np.where((T_all == T_plot) & (P_all == P_plot))[0]
        
    T = [T_all[j] for j in i_plot]
    P = [P_all[j] for j in i_plot]
    
    wvn_labfit_iter = np.concatenate([wvn_pure[i] for i in i_plot])
    trans_labfit_iter = np.concatenate([trans_pure[i] for i in i_plot])
    res_SDVP_iter = np.concatenate([res_SDVP_raw[i] for i in i_plot])
    res_VP_iter = np.concatenate([res_VP_raw[i] for i in i_plot])
    res_HT_iter = np.concatenate([res_HT_raw[i] for i in i_plot])
    
    
    [istart, istop] = td.bandwidth_select_td(wvn_labfit_iter, wvn2_data, max_prime_factor=500, print_value=False)
    
    wvn.append(wvn_labfit_iter[istart:istop])
    trans.append(trans_labfit_iter[istart:istop]/100)
    res_SDVP.append(res_SDVP_iter[istart:istop]/100)
    res_VP.append(res_VP_iter[istart:istop]/100)
    res_HT.append(res_HT_iter[istart:istop]/100) 

    # plt.plot(wvn[-1], trans[-1], label='updated')
    # plt.plot(wvn[-1], res_updated[-1], label='og')
    # plt.plot(wvn[-1], res_HT[-1], label='HT')

    RMS_HT[i] = np.sqrt(np.mean(res_HT[-1][20000:-20000]**2))
    RMS_SDVP[i] = np.sqrt(np.mean(res_SDVP[-1][20000:-20000]**2))
    RMS_VP[i] = np.sqrt(np.mean(res_VP[-1][20000:-20000]**2))
    Prms[i] = np.mean(P)
# plt.legend()

# plt.figure()
# plt.plot([x*0.0001 for x in Prms], label='pressure')
# plt.plot(RMS_HT, label='HT')
# plt.plot(RMS_updated, label='updated')
# plt.legend()



#%%

wide = [6615-25, 7650+25]
# narrow1 = [7046.35, 7047.93]
narrow1 = [7070.395, 7071.838]
narrow2 = [6717.89, 6719.36]



if d_type == 'pure': 
    
    y_lim_top = [0.351,1.12]
    y_lim_bottom = [-0.075,0.075]
    
    y_lim_top_narrow = [0.951,1.01]
    y_lim_bottom_narrow = [-0.034,0.034]

elif d_type == 'air': 
    
    y_lim_top = [0.63,1.08]
    y_lim_bottom = [-0.04,0.04]
    
    y_lim_top_narrow = [0.965,1.007]
    y_lim_bottom_narrow = [-0.019,0.019]


linewidth = 1

offset1 = 0.03
offset0 = offset1*25

# colors = ['#0028ff','#0080af','#117d11','#be961e', '#ff4000','#ff0000',      '#e6ab02', '#fee9ac']
colors = ['#ff0000','#d95f02', '#1b9e77','#4c7c17',       '#514c8e','#dfdeed',         '#e6ab02', '#fee9ac']
# colors = ['#FFD700','#FF7F50','#EE82EE','#4169E1', '#00BFFF','#00FFFF',     '#e6ab02']
# colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']



num_files = 6


which_files_partial = which_files[:num_files][::-1]


fig = plt.figure(figsize=(14.4, 6))

gs_right = GridSpec(4, 3, width_ratios = [3,1,1], height_ratios=[3,1,1,1], hspace=0.02, wspace=0.06) # rows, columns
gs = GridSpec(4, 3, width_ratios = [3,1,1], height_ratios=[3,1,1,1], hspace=0.02, wspace=0.005) # rows, columns

for which_file in which_files_partial: 
    
    i = which_files.index(which_file)
    
    ax00 = fig.add_subplot(gs[0,0]) # First row, first column
    ax00.plot(wvn[i],trans[i], color=colors[i], label='{} H$_{{2}}$O{} at {}orr'.format(y_h2o, post_label, which_file), 
              linewidth=linewidth)
    ax00.legend(loc = 'lower right', framealpha=1, edgecolor='black', fontsize=10, labelspacing=0)   
   
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00) # Second row, first column
    ax10.plot(wvn[i],res_HT[i], color=colors[i+1], label=which_file, 
              linewidth=linewidth)
    
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00) # Second row, first column
    ax20.plot(wvn[i],res_VP[i], color=colors[i+2], label=which_file, 
              linewidth=linewidth)
    
    ax30 = fig.add_subplot(gs[3,0], sharex=ax00) # Second row, first column
    ax30.plot(wvn[i],res_SDVP[i], color=colors[i+3], label=which_file, 
              linewidth=linewidth)
    
    
    
    # second column
    ax01 = fig.add_subplot(gs[0,1]) # First row, second column
    ax01.plot(wvn[i],trans[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
   
    ax11 = fig.add_subplot(gs[1,1], sharex=ax01) # Second row, second column
    ax11.plot(wvn[i],res_HT[i], color=colors[i+1], label=which_file, 
              linewidth=linewidth)
    
    ax21 = fig.add_subplot(gs[2,1], sharex=ax01) # Second row, second column
    ax21.plot(wvn[i],res_VP[i], color=colors[i+2], label=which_file, 
              linewidth=linewidth)
    
    ax31 = fig.add_subplot(gs[3,1], sharex=ax01) # Second row, second column
    ax31.plot(wvn[i],res_SDVP[i], color=colors[i+3], label=which_file, 
              linewidth=linewidth)
    
    
    
    # third column
    ax02 = fig.add_subplot(gs_right[0,2]) # First row, third column
    ax02.plot(wvn[i],trans[i], color=colors[i], label=which_file, 
              linewidth=linewidth)
  
   
    ax12 = fig.add_subplot(gs_right[1,2], sharex=ax02) # Second row, third column
    ax12.plot(wvn[i],res_HT[i], color=colors[i+1], label=which_file, 
              linewidth=linewidth)
    
    ax22 = fig.add_subplot(gs_right[2,2], sharex=ax02) # Second row, third column
    ax22.plot(wvn[i],res_VP[i], color=colors[i+2], label=which_file, 
              linewidth=linewidth)
    
    ax32 = fig.add_subplot(gs_right[3,2], sharex=ax02) # Second row, third column
    ax32.plot(wvn[i],res_SDVP[i], color=colors[i+3], label=which_file, 
              linewidth=linewidth)
    
    
    ax00.axvline(narrow1[0]-offset0, linewidth=1, color=colors[-2])
    ax00.axvline(narrow1[1]+offset0, linewidth=1, color=colors[-2])
    ax10.axvline(narrow1[0]-offset0, linewidth=1, color=colors[-2])
    ax10.axvline(narrow1[1]+offset0, linewidth=1, color=colors[-2])
    ax20.axvline(narrow1[0]-offset0, linewidth=1, color=colors[-2])
    ax20.axvline(narrow1[1]+offset0, linewidth=1, color=colors[-2])
    ax30.axvline(narrow1[0]-offset0, linewidth=1, color=colors[-2])
    ax30.axvline(narrow1[1]+offset0, linewidth=1, color=colors[-2])
    
    ax00.axvline(narrow2[0]-offset0, linewidth=1, color=colors[-4])
    ax00.axvline(narrow2[1]+offset0, linewidth=1, color=colors[-4])
    ax10.axvline(narrow2[0]-offset0, linewidth=1, color=colors[-4])
    ax10.axvline(narrow2[1]+offset0, linewidth=1, color=colors[-4])
    ax20.axvline(narrow2[0]-offset0, linewidth=1, color=colors[-4])
    ax20.axvline(narrow2[1]+offset0, linewidth=1, color=colors[-4])
    ax30.axvline(narrow2[0]-offset0, linewidth=1, color=colors[-4])
    ax30.axvline(narrow2[1]+offset0, linewidth=1, color=colors[-4])
       
    
#%% arrows pointing to inset

if d_type == 'pure': 
    ax00.arrow(narrow1[1], 0.42, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
    ax00.arrow(narrow2[1], 0.84, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')

elif d_type == 'air': 
    ax00.arrow(narrow1[1], 0.67, 75, 0, length_includes_head=True, head_width=0.03, head_length=30, color='k')
    ax00.arrow(narrow2[1], 0.84, 75, 0, length_includes_head=True, head_width=0.03, head_length=30, color='k')


#%% set axis
ax00.set_xlim(wide)

ax00.set_ylim(y_lim_top)
ax10.set_ylim(y_lim_bottom)
ax20.set_ylim(y_lim_bottom)
ax30.set_ylim(y_lim_bottom)


ax01.set_xlim(narrow1)

ax01.set_ylim(y_lim_top)
ax11.set_ylim(y_lim_bottom)
ax21.set_ylim(y_lim_bottom)
ax31.set_ylim(y_lim_bottom)

ax02.set_xlim(narrow2)

ax02.set_ylim(y_lim_top_narrow)
ax12.set_ylim(y_lim_bottom_narrow)
ax22.set_ylim(y_lim_bottom_narrow)
ax32.set_ylim(y_lim_bottom_narrow)

#%%  remove x label on upper plots (mostly covered)
plt.setp(ax00.get_xticklabels(), visible=False) 
plt.setp(ax10.get_xticklabels(), visible=False)
plt.setp(ax20.get_xticklabels(), visible=False)

plt.setp(ax01.get_xticklabels(), visible=False)
plt.setp(ax11.get_xticklabels(), visible=False)
plt.setp(ax21.get_xticklabels(), visible=False)

plt.setp(ax02.get_xticklabels(), visible=False)
plt.setp(ax12.get_xticklabels(), visible=False)
plt.setp(ax22.get_xticklabels(), visible=False)


#%%  remove y label on middle plots (might be a bad idea)
plt.setp(ax01.get_yticklabels(), visible=False) 
plt.setp(ax11.get_yticklabels(), visible=False)
plt.setp(ax21.get_yticklabels(), visible=False)
plt.setp(ax31.get_yticklabels(), visible=False)

#%% move zoomed y labels to the right
ax02.yaxis.set_label_position("right")
ax02.yaxis.tick_right()
ax12.yaxis.set_label_position("right")
ax12.yaxis.tick_right()
ax22.yaxis.set_label_position("right")
ax22.yaxis.tick_right()
ax32.yaxis.set_label_position("right")
ax32.yaxis.tick_right()

# %% add ticks and minor ticks all over
ax00.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=True)
ax00.xaxis.set_minor_locator(AutoMinorLocator(10))
ax00.yaxis.set_minor_locator(AutoMinorLocator(10))

ax10.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=True)
ax10.xaxis.set_minor_locator(AutoMinorLocator(5))
ax10.yaxis.set_minor_locator(AutoMinorLocator(5))

ax20.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=True)
ax20.xaxis.set_minor_locator(AutoMinorLocator(5))
ax20.yaxis.set_minor_locator(AutoMinorLocator(5))

ax30.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=True)
ax30.xaxis.set_minor_locator(AutoMinorLocator(5))
ax30.yaxis.set_minor_locator(AutoMinorLocator(5))


ax01.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax01.xaxis.set_minor_locator(AutoMinorLocator(10))
ax01.yaxis.set_minor_locator(AutoMinorLocator(10))

ax11.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax11.xaxis.set_minor_locator(AutoMinorLocator(5))
ax11.yaxis.set_minor_locator(AutoMinorLocator(5))

ax21.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax21.xaxis.set_minor_locator(AutoMinorLocator(5))
ax21.yaxis.set_minor_locator(AutoMinorLocator(5))

ax31.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=True, right=False)
ax31.xaxis.set_minor_locator(AutoMinorLocator(5))
ax31.yaxis.set_minor_locator(AutoMinorLocator(5))


ax02.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax02.xaxis.set_minor_locator(AutoMinorLocator(5))
ax02.yaxis.set_minor_locator(AutoMinorLocator(5))

ax12.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax12.xaxis.set_minor_locator(AutoMinorLocator(5))
ax12.yaxis.set_minor_locator(AutoMinorLocator(5))

ax22.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax22.xaxis.set_minor_locator(AutoMinorLocator(5))
ax22.yaxis.set_minor_locator(AutoMinorLocator(5))

ax32.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax32.xaxis.set_minor_locator(AutoMinorLocator(5))
ax32.yaxis.set_minor_locator(AutoMinorLocator(5))


#%% shading to highlight zoomed region
alpha = 1

ax00.axvspan(narrow1[0]-offset0, narrow1[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax10.axvspan(narrow1[0]-offset0, narrow1[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax20.axvspan(narrow1[0]-offset0, narrow1[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax30.axvspan(narrow1[0]-offset0, narrow1[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)

ax01.axvspan(narrow1[0]+offset1, narrow1[1]-offset1, alpha=alpha, color=colors[-1], zorder=0)
ax11.axvspan(narrow1[0]+offset1, narrow1[1]-offset1, alpha=alpha, color=colors[-1], zorder=0)
ax21.axvspan(narrow1[0]+offset1, narrow1[1]-offset1, alpha=alpha, color=colors[-1], zorder=0)
ax31.axvspan(narrow1[0]+offset1, narrow1[1]-offset1, alpha=alpha, color=colors[-1], zorder=0)

ax00.axvspan(narrow2[0]-offset0, narrow2[1]+offset0, alpha=alpha, color=colors[-3], zorder=0)
ax10.axvspan(narrow2[0]-offset0, narrow2[1]+offset0, alpha=alpha, color=colors[-3], zorder=0)
ax20.axvspan(narrow2[0]-offset0, narrow2[1]+offset0, alpha=alpha, color=colors[-3], zorder=0)
ax30.axvspan(narrow2[0]-offset0, narrow2[1]+offset0, alpha=alpha, color=colors[-3], zorder=0)

ax02.axvspan(narrow2[0]+offset1, narrow2[1]-offset1, alpha=alpha, color=colors[-3], zorder=0)
ax12.axvspan(narrow2[0]+offset1, narrow2[1]-offset1, alpha=alpha, color=colors[-3], zorder=0)
ax22.axvspan(narrow2[0]+offset1, narrow2[1]-offset1, alpha=alpha, color=colors[-3], zorder=0)
ax32.axvspan(narrow2[0]+offset1, narrow2[1]-offset1, alpha=alpha, color=colors[-3], zorder=0)


#%% labels
ax20.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax21.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax22.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')


ax00.set_ylabel('Measured Transmission\n')
ax10.set_ylabel('Meas-HT')
ax20.set_ylabel('Meas-\nT.W. (VP)')
ax30.set_ylabel('Meas-T.W.\n(SDVP)')

ax02.set_ylabel('Measured Transmission\n')
ax12.set_ylabel('Meas-HT')
ax22.set_ylabel('Meas-\nT.W. (VP)')
ax32.set_ylabel('Meas-T.W.\n(SDVP)')

#%%

if d_type == 'pure': 

    h0 = 0.015
    h1 = 0.03
    v0 = 0.9
    
    hb = 0.50
    vb = 0.13
    
    hc = 0.18
    vc = 0.56

elif d_type == 'air': 

    h0 = 0.015
    h1 = 0.03
    v0 = 0.9
    
    hb = 0.50
    vb = 0.13
    
    hc = 0.18
    vc = 0.53

ax00.text(h0, v0, "A", fontweight="bold", fontsize=12, transform=ax00.transAxes)
ax01.text(h1, v0, "B", fontweight="bold", fontsize=12, transform=ax01.transAxes)
ax02.text(h1, v0, "C", fontweight="bold", fontsize=12, transform=ax02.transAxes)

ax00.text(hb, vb, "(B)", fontsize=12, transform=ax00.transAxes)
ax00.text(hc, vc, "(C)", fontsize=12, transform=ax00.transAxes)

ax02.text(0.18, 0.85, "*", fontweight="bold", fontsize=12, transform=ax02.transAxes)


#%% save it

plt.savefig(r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\plots\7-1 big residual {}.svg'.format(d_type), bbox_inches='tight')

