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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

import pldspectrapy as pld
import td_support as td

import numpy as np

#%% setup things

d_type = 'air' # 'pure' or 'air'
air_og = False

wvn2_processing = [6500, 7800] # range used when processing the data
wvn2_data = [6615, 7650] # where there is actually useful data that we would want to include

if d_type == 'pure': 
    which_file = '1300 K 16 T'
    y_h2o = '2%'
    post_label = ''
    which_vacuum = 25 # vacuum scans that correspond to the file above
elif d_type == 'air': 
    if air_og:
        which_file = '1300 K 600 T'
        y_h2o = '2%'
        post_label = ' in Air'
        which_vacuum = 25 # vacuum scans that correspond to the file above
    else: 
        which_file = '1100 K 600 T'
        y_h2o = '2%'
        post_label = ' in Air'
        which_vacuum = 19 # vacuum scans that correspond to the file above

#%% load in transmission data (model from labfit results)

# load in labfit stuff (transmission, wvn, residuals before and after, conditions)
# d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'
d_sceg = r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\data - sceg'



if d_type == 'pure': f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
elif d_type == 'air': f = open(os.path.join(d_sceg,'spectra_air.pckl'), 'rb')

[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure, res_sd0] = pickle.load(f)
f.close()

[T_all, P_all] = np.asarray([T_pure, P_pure])

if d_type == 'air': P_all = np.round(P_all/10,0)*10 # for air-water to round pressures to nearest 10's

T_plot = int(which_file.split()[0])
P_plot = int(which_file.split()[2])

i_plot = np.where((T_all == T_plot) & (P_all == P_plot))[0]
    
T = [T_all[i] for i in i_plot]
P = [P_all[i] for i in i_plot]

wvn_labfit_all = np.concatenate([wvn_pure[i] for i in i_plot])
trans_labfit_all = np.concatenate([trans_pure[i] for i in i_plot])
res_updated_all = np.concatenate([res_pure[i] for i in i_plot])
res_og_all = np.concatenate([res_og_pure[i] for i in i_plot])


#%% load in transmission data (vacuum normalized from bg subtract)

if d_type == 'pure': d_measurement = r'E:\water database\data - 2021-08\pure water'
elif d_type == 'air': d_measurement = r'E:\water database\data - 2021-08\air water'
f = open(os.path.join(d_measurement, which_file+' bg subtraction.pckl'), 'rb')
# [meas_trans_bg, meas_trans_bl, wvn, T, P, y_h2o, pathlength, favg, fitresults_all, model_trans_fit2020, model_trans_fit2016, model_trans_fitPaul]
[transmission, _, wvn_process, _, _, _, _, _, _, model, _, _] = pickle.load(f)
f.close()

#%% load in raw measurement data

d_ref = True # there was a reference channel

f = open(os.path.join(d_measurement, which_file+' pre.pckl'), 'rb')
# [measurement_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, trans_raw_ref, trans_snip_ref]
if d_ref: [measurement_w_spike, _, _, _, frep1, frep2, ppig, fopt, _, _] = pickle.load(f)
else: [measurement_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt] = pickle.load(f)
f.close()

# calculate extended wavenumber axis of raw measuremnt
spike_location_expected = 13979
measurement = measurement_w_spike[spike_location_expected:-1] / max(measurement_w_spike[spike_location_expected:-1]) # normalize max to 1


hz2cm = 1 / pld.SPEED_OF_LIGHT / 100 # conversion from MHz to cm-1

fclocking = frep2 # really it was comb 1, but for some reason this is what silmaril data wants
favg = np.mean([frep1,frep2])

nyq_range = ppig*2 * fclocking
nyq_num = np.round(1e6 / (ppig*2))

fcw = nyq_range * nyq_num

nyq_start = fcw + fopt

# if nyq_side == 1:
nyq_stop = nyq_range * (nyq_num + 0.5) + fopt # Nyquist window 1.0 - 1.5
wvn_raw = np.arange(nyq_start, nyq_stop, favg) * hz2cm
wvn_raw = wvn_raw[:len(measurement)]
        


#%% load in vacuum scans

d_vacuum = r'E:\water database\data - 2021-08\vacuum scans'

f = open(os.path.join(d_vacuum, 'plot stuff 1 0.030 with 2Ts.pckl'), 'rb')
# [meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final]
[meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final] = pickle.load(f)
f.close() 

vacuum_raw = meas_raw_all_final[which_vacuum]
vacuum_h2o = meas_bg_all_final[which_vacuum]
vacuum_smooth = meas_filt_all_final[which_vacuum]


#%% plot and verify this is what you want

# plt.plot(wvn_process, transmission) # verify this is the data you want
# plt.plot(wvn_process, model) # verify this is the data you want
# plt.plot(wvn_raw, measurement) # verify this is the data you want


[istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_data, max_prime_factor=500, print_value=False)
meas_data = measurement[istart:istop] / max(measurement[istart:istop])

[istart, istop] = td.bandwidth_select_td(wvn_process, wvn2_data, max_prime_factor=500, print_value=False)
vac_raw_data = vacuum_raw[istart:istop] / max(vacuum_smooth[istart:istop])
vac_h2o_data = vacuum_h2o[istart:istop] / max(vacuum_smooth[istart:istop])
vac_smooth_data = vacuum_smooth[istart:istop] / max(vacuum_smooth[istart:istop])

trans_data = meas_data / vac_smooth_data

wvn_data = wvn_process[istart:istop]

[istart, istop] = td.bandwidth_select_td(wvn_labfit_all, wvn2_data, max_prime_factor=500, print_value=False)

wvn_labfit = wvn_labfit_all[istart:istop] # this should be the same as wvn_data (not sure why it isn't)
trans_labfit = trans_labfit_all[istart:istop] / 100 + 0.0001 # for the inset plot (the transmissino is still rising at the edge)
res_updated = res_updated_all[istart:istop] 
res_og = res_og_all[istart:istop] 


# plt.plot(wvn_data, trans_data) # verify this is the data you want
# plt.plot(wvn_data, vac_raw_data) # verify this is the data you want
# plt.plot(wvn_data, vac_h2o_data) # verify this is the data you want
# plt.plot(wvn_data, vac_smooth_data) # verify this is the data you want
# plt.plot(wvn_data, meas_data) # verify this is the data you want

# asdfasdfsd

#%%

fig = plt.figure(figsize=(14.4, 7))

gs = GridSpec(3, 2, width_ratios=[2, 1], hspace=0.015, wspace=0.005) # rows, columns

wide = [6615-50, 7650+25]

if d_type == 'pure': 
    narrow = [7094.39, 7096.19]
    
if d_type == 'air': 
    narrow = [6962.66, 6963.94]

linewidth = 1

offset1 = 0.05
offset0 = 0.05*25

# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=6 
# (#fee9ac is #e6ab02 lightened, same for #e7298a -> #f5a9d0, #66a61e was darkened to #4c7c17, same for #7570b3 to #514c8e)
colors = ['#d95f02','#1b9e77','k','#514c8e','#f5a9d0', '#4c7c17','#e6ab02', '#fee9ac']

ax00 = fig.add_subplot(gs[0,0]) # First row, first column
ax00.axvline(narrow[0]-offset0, linewidth=1, color=colors[-2])
ax00.axvline(narrow[1]+offset0, linewidth=1, color=colors[-2])
ax00.plot(wvn_data,vac_raw_data, color=colors[0], label='Baseline - Optical Cell at {} K <1mTorr'.format(which_file.split()[0]), 
          linewidth=linewidth)
ax00.plot(wvn_data,vac_h2o_data, color=colors[1], label='Baseline - Background H$_{2}$O Removed', 
          linewidth=linewidth)
ax00.plot(wvn_data,vac_smooth_data, color=colors[2], label='Baseline - Low-pass Filtered', 
          linewidth=linewidth*2)
ax00.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)

ax01 = fig.add_subplot(gs[0,1]) # First row, second column
ax01.plot(wvn_data,vac_raw_data, color=colors[0], 
          linewidth=linewidth)
ax01.plot(wvn_data,vac_h2o_data, color=colors[1], 
          linewidth=linewidth)
ax01.plot(wvn_data,vac_smooth_data, color=colors[2], 
          linewidth=linewidth*2)


ax10 = fig.add_subplot(gs[1,0], sharex = ax00) # Second row, first column
ax10.axvline(narrow[0]-offset0, linewidth=1, color=colors[-2])
ax10.axvline(narrow[1]+offset0, linewidth=1, color=colors[-2])

ax10.plot(wvn_data, meas_data, color=colors[3], linewidth=linewidth,
          label='{} H$_{{2}}$O{} at {}orr'.format(y_h2o, post_label, which_file))
ax10.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)

ax11 = fig.add_subplot(gs[1,1], sharex = ax01) # Second row, second column
ax11.plot(wvn_data, meas_data, color=colors[3], 
          linewidth=linewidth)


ax20 = fig.add_subplot(gs[2,0], sharex = ax00) # Third row, first column
ax20.axvline(narrow[0]-offset0, linewidth=1, color=colors[-2])
ax20.axvline(narrow[1]+offset0, linewidth=1, color=colors[-2])
ax20.plot(wvn_data, trans_data, color=colors[4], linewidth=linewidth,
          label='{} H$_{{2}}$O{} at {}orr - Normalized by Filtered Baseline'.format(y_h2o, post_label, which_file))
ax20.plot(wvn_labfit, trans_labfit, color=colors[5], linewidth=linewidth,
          label='{} H$_{{2}}$O{} at {}orr - Normalized by Baseline and Chebyshevs'.format(y_h2o, post_label, which_file))
ax20.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=10)

ax21 = fig.add_subplot(gs[2,1], sharex = ax01) # Third row, second column
ax21.plot(wvn_data, trans_data, color=colors[4], 
          linewidth=linewidth)
ax21.plot(wvn_labfit, trans_labfit, color=colors[5],
          linewidth=linewidth)

#%% noise plot inset 

if d_type == 'pure': 
    ax21ins = inset_axes(ax21, width='30%', height='40%', loc='lower left', bbox_to_anchor=(0.22,0.1,1.2,1.2), bbox_transform=ax21.transAxes)
if d_type == 'air': 
    ax21ins = inset_axes(ax21, width='30%', height='40%', loc='lower left', bbox_to_anchor=(0.55,0.03,1.2,1.2), bbox_transform=ax21.transAxes)

ax21ins.plot(wvn_labfit, trans_labfit, color=colors[5], 
             linewidth=linewidth)

if d_type == 'pure': 
    ax21ins.axis([7094.885, 7095.20, 0.9985, 1.00025])

elif d_type == 'air': 
    # ax21ins.axis([6962.45, 6962.735, 0.9987, 1.00065])
    ax21ins.axis([6963.48, 6963.695, 0.9984, 1.00065])


patch, pp1,pp2 = mark_inset(ax21, ax21ins, loc1=1, loc2=2, fc='none', ec='k', zorder=5)
pp1.loc2 = 4
    
ax21ins.xaxis.set_visible(False)

if d_type == 'pure': 

    ax21ins.yaxis.set_label_position("right")
    ax21ins.yaxis.tick_right()
    ax21ins.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax21ins.text(0.68, 0.2, "noise\n floor", fontsize=10, transform=ax21ins.transAxes)

if d_type == 'air': 

    ax21ins.yaxis.set_label_position("left")
    ax21ins.yaxis.tick_left()
    ax21ins.yaxis.set_minor_locator(AutoMinorLocator(5))

    if air_og: ax21ins.text(0.4, 0.2, "noise floor", fontsize=10, transform=ax21ins.transAxes)
    else: ax21ins.text(0.4, 0.1, "noise floor", fontsize=10, transform=ax21ins.transAxes)


#%% 1300 K inset

# ax00ins = inset_axes(ax00, width='15%', height='30%', loc='upper left', bbox_to_anchor=(0.82,-0.01,1,1), bbox_transform=ax00.transAxes)

# ax00ins.axvspan(narrow[0], narrow[1], color='#ff5d00', zorder=0)

# # ax00ins.plot(wvn_data, trans_data, color=colors[4])
# ax00ins.axis([7094.88, 7095.20, 1.0111, 1.0132])

# # patch, pp1,pp2 = mark_inset(ax00, ax00ins, loc1=1, loc2=2, fc='none', ec='k', zorder=0)
# # pp1.loc2 = 4

# ax00ins.xaxis.set_visible(False)
# ax00ins.yaxis.set_visible(False)

# ax00ins.text(0.22, 0.3, "1300 K\n  (ref)", fontweight="bold", fontsize=8, transform=ax00ins.transAxes)


#%% arrows pointing to inset

if d_type == 'pure': 

    ax00.arrow(narrow[1], 0.525, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
    ax10.arrow(narrow[1], 0.27, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
    ax20.arrow(narrow[1], 0.435, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')

if d_type == 'air': 

    ax00.arrow(narrow[1], 0.55, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
    ax10.arrow(narrow[1], 0.3, 75, 0, length_includes_head=True, head_width=0.05, head_length=30, color='k')
    if air_og: ax20.arrow(narrow[1], 0.68, 75, 0, length_includes_head=True, head_width=0.03, head_length=30, color='k')
    else: ax20.arrow(narrow[1], 0.6, 75, 0, length_includes_head=True, head_width=0.03, head_length=30, color='k')

#%% set axis

ax00.set_xlim(wide)
ax01.set_xlim(narrow)


if d_type == 'pure': 
    
    ax01.set_ylim([0.861, 0.886])
    ax11.set_ylim([0.35, 0.99])
    ax21.set_ylim([0.41, 1.099])
    ax00.set_ylim([0,1.1])
    ax10.set_ylim([0,1.1])
    ax20.set_ylim([0,1.1])

elif d_type == 'air': 

    ax21.xaxis.set_ticks(np.arange(6962.8, 6964.0, 0.3))

    ax00.set_ylim([0,1.1])
    ax10.set_ylim([0,1.1])

    if air_og: 
        ax01.set_ylim([0.953, 0.959])
        ax11.set_ylim([0.901, 0.938])
        ax21.set_ylim([0.96, 1.01])
        
        ax20.set_ylim([0.45,1.06])
    
    else: 
        ax01.set_ylim([0.988, 0.993])
        ax11.set_ylim([0.963, 0.999])
        ax21.set_ylim([0.969, 1.005])
        
        ax20.set_ylim([0.3,1.06])


#%%  remove x label on upper plots (mostly covered)
plt.setp(ax00.get_xticklabels(), visible=False) 
plt.setp(ax01.get_xticklabels(), visible=False)
plt.setp(ax10.get_xticklabels(), visible=False)
plt.setp(ax11.get_xticklabels(), visible=False)


#%% move zoomed y labels to the right
ax01.yaxis.set_label_position("right")
ax01.yaxis.tick_right()
ax11.yaxis.set_label_position("right")
ax11.yaxis.tick_right()
ax21.yaxis.set_label_position("right")
ax21.yaxis.tick_right()


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

ax01.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax01.xaxis.set_minor_locator(AutoMinorLocator(5))
ax01.yaxis.set_minor_locator(AutoMinorLocator(5))

ax11.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax11.xaxis.set_minor_locator(AutoMinorLocator(5))
ax11.yaxis.set_minor_locator(AutoMinorLocator(5))

ax21.tick_params(axis='both', which='both', direction='in', top=False, bottom=True, left=False, right=True)
ax21.xaxis.set_minor_locator(AutoMinorLocator(5))
ax21.yaxis.set_minor_locator(AutoMinorLocator(5))
    
if d_type == 'air': ax21.xaxis.set_minor_locator(MultipleLocator(0.05))


#%% shading to highlight zoomed region
alpha = 1

ax00.axvspan(narrow[0]-offset0, narrow[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax10.axvspan(narrow[0]-offset0, narrow[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax20.axvspan(narrow[0]-offset0, narrow[1]+offset0, alpha=alpha, color=colors[-1], zorder=0)
ax01.axvspan(narrow[0]+offset1, narrow[1]-offset1*1.2, alpha=alpha, color=colors[-1], zorder=0)
ax11.axvspan(narrow[0]+offset1, narrow[1]-offset1*1.2, alpha=alpha, color=colors[-1], zorder=0)
ax21.axvspan(narrow[0]+offset1, narrow[1]-offset1*1.2, alpha=alpha, color=colors[-1], zorder=0)


#%% labels
ax21.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax20.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

ax00.set_ylabel('Intensity (arb.)')
ax10.set_ylabel('Intensity (arb.)')
ax20.set_ylabel('Transmission')


#%%


if d_type == 'pure': 
    
    h0 = 0.02
    h1 = 0.05
    v0 = 0.88
    v1 = 0.88
    
    ax00.text(0.522, 0.53, "(B)", fontsize=12, transform=ax00.transAxes)
    ax10.text(0.522, 0.3, "(D)", fontsize=12, transform=ax10.transAxes)
    ax20.text(0.522, 0.45, "(F)", fontsize=12, transform=ax20.transAxes)


    ax00.text(h0, v1, "A", fontweight="bold", fontsize=12, transform=ax00.transAxes)   
    ax01.text(h1, v1, "B", fontweight="bold", fontsize=12, transform=ax01.transAxes)
    
    ax10.text(h0, v1, "C", fontweight="bold", fontsize=12, transform=ax10.transAxes)    
    ax11.text(h1, v1, "D", fontweight="bold", fontsize=12, transform=ax11.transAxes)
    
    ax20.text(h0, v0, "E", fontweight="bold", fontsize=12, transform=ax20.transAxes)   
    ax21.text(h1, v1, "F", fontweight="bold", fontsize=12, transform=ax21.transAxes)
    
elif d_type == 'air': 
    
    h0 = 0.02
    h1 = 0.07
    v0 = 0.87
    v1 = 0.9
    
    if air_og: 
        h2 = 0.405
        vf = 0.425
    else: 
        h2 = 0.402
        vf = 0.445
    
    ax00.text(h2, 0.55, "(B)", fontsize=12, transform=ax00.transAxes)
    ax10.text(h2, 0.32, "(D)", fontsize=12, transform=ax10.transAxes)
    ax20.text(h2, vf, "(F)", fontsize=12, transform=ax20.transAxes)

    ax00.text(h0, v1, "A", fontweight="bold", fontsize=12, transform=ax00.transAxes)   
    ax01.text(h1, v1, "B", fontweight="bold", fontsize=12, transform=ax01.transAxes)
    
    ax10.text(h0, v1, "C", fontweight="bold", fontsize=12, transform=ax10.transAxes)    
    ax11.text(h1, v1, "D", fontweight="bold", fontsize=12, transform=ax11.transAxes)
    
    ax20.text(h0, v0, "E", fontweight="bold", fontsize=12, transform=ax20.transAxes)   
    ax21.text(h1, v1, "F", fontweight="bold", fontsize=12, transform=ax21.transAxes)



#%% save it



if d_type == 'pure':
    plt.savefig(r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\plots\7-1big.svg',bbox_inches='tight')
elif d_type == 'air':
    plt.savefig(r'C:\Users\silmaril\Documents\from scott - making silmaril a water computer\Silmaril-to-LabFit-Processing-Scripts\plots\7-1big air.svg',bbox_inches='tight')
