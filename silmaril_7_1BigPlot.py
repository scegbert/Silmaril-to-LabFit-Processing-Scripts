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

which_file = '1300 K 16 T'
which_vacuum = 25 # vacuum scans that correspond to the file above


#%% load in transmission data (model from labfit results)

# load in labfit stuff (transmission, wvn, residuals before and after, conditions)
d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure] = pickle.load(f)
f.close()

transmission_labfit = trans_pure

#%% load in transmission data (vacuum normalized from bg subtract)

d_measurement = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'
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

d_vacuum = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'

f = open(os.path.join(d_vacuum, 'plot stuff 1 0.030 with 2Ts.pckl'), 'rb')
# [meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final]
[meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final] = pickle.load(f)
f.close() 

vacuum_raw = meas_spike_all_final[which_vacuum]
vacuum_h2o = meas_bg_all_final[which_vacuum]
vacuum_smooth = meas_filt_all_final[which_vacuum]


#%% plot and verify this is what you want

# plt.plot(wvn_process, transmission) # verify this is the data you want
# plt.plot(wvn_process, model) # verify this is the data you want
# plt.plot(wvn_raw, measurement) # verify this is the data you want


[istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_data, max_prime_factor=500, print_value=False)
meas_data = measurement[istart:istop]

[istart, istop] = td.bandwidth_select_td(wvn_process, wvn2_data, max_prime_factor=500, print_value=False)
vac_raw_data = vacuum_raw[istart:istop]
vac_h2o_data = vacuum_h2o[istart:istop]
vac_smooth_data = vacuum_smooth[istart:istop]
trans_data = transmission[istart:istop]


wvn_data = wvn_process[istart:istop]


plt.plot(wvn_data, trans_data) # verify this is the data you want
plt.plot(wvn_data, vac_raw_data) # verify this is the data you want
plt.plot(wvn_data, vac_h2o_data) # verify this is the data you want
plt.plot(wvn_data, vac_smooth_data) # verify this is the data you want
plt.plot(wvn_data, meas_data) # verify this is the data you want




#%%

fig = plt.figure(figsize=(10, 7.5))

gs = GridSpec(3, 2, width_ratios=[2, 1], hspace=0.015, wspace=0.005) # rows, columns

wide = [6615-14, 7650+14]
narrow = [7094.249, 7096.01] # [7093.9, 7096.1] # [7070.62, 7071.78] # [7089.5, 7100.5]
inset21 = [7094.81, 7095, 0.9985, 1.0019]

colors = ['dodgerblue', 'firebrick', 'darkorange', 'dodgerblue', 'firebrick', 'moccasin']

ax00 = fig.add_subplot(gs[0,0]) # First row, first column
ax00.axvline(narrow[0], linewidth=1, color=colors[2])
ax00.axvline(narrow[1], linewidth=1, color=colors[2])
ax00.plot(wvn_data,vac_raw_data, color=colors[0], label='Laser Baseline (Gas Cell at Vacuum)')
ax00.plot(wvn_data,vac_h2o_data, color=colors[1], label='Baseline without Background $\mathregular{H_2O}$')
ax00.plot(wvn_data,vac_smooth_data, color=colors[2], label='Baseline after Low-pass Filter')
ax00.legend(loc = 'upper right', framealpha=1, edgecolor='black')

ax01 = fig.add_subplot(gs[0,1]) # First row, second column
ax01.plot(wvn_data,vac_raw_data, color=colors[0])
ax01.plot(wvn_data,vac_h2o_data, color=colors[1])
ax01.plot(wvn_data,vac_smooth_data, color=colors[2], linewidth=2)


ax10 = fig.add_subplot(gs[1,0], sharex = ax00) # Second row, first column
ax10.axvline(narrow[0], linewidth=1, color=colors[2])
ax10.axvline(narrow[1], linewidth=1, color=colors[2])
ax10.plot(wvn, meas, color=colors[3], label='Pure $\mathregular{H_2O}$ at 1100 K 16 T Measurement')
ax10.legend(loc = 'upper right', framealpha=1, edgecolor='black')

ax11 = fig.add_subplot(gs[1,1], sharex = ax01) # Second row, second column
ax11.plot(wvn, meas, color=colors[3])


ax20 = fig.add_subplot(gs[2,0], sharex = ax00) # Third row, first column
ax20.axvline(narrow[0], linewidth=1, color=colors[2])
ax20.axvline(narrow[1], linewidth=1, color=colors[2])
ax20.plot(wavenumbers[which_norm], transmission_bl[which_norm], color=colors[4], label='Pure $\mathregular{H_2O}$ at 1100 K 16 T, Normalized by Baseline')
ax20.legend(loc = 'lower left', framealpha=1, edgecolor='black')

ax21 = fig.add_subplot(gs[2,1], sharex = ax01) # Third row, second column
ax21.plot(wavenumbers[which_norm], transmission_bl[which_norm], color=colors[4])

#%% noise plot inset 

ax21ins = inset_axes(ax21, width='30%', height='40%', loc='lower left', bbox_to_anchor=(0.35,0.1,1.2,1.2), bbox_transform=ax21.transAxes)
ax21ins.plot(wavenumbers[which_norm], transmission_bl[which_norm], color=colors[4])
ax21ins.axis(inset21)

patch, pp1,pp2 = mark_inset(ax21, ax21ins, loc1=1, loc2=2, fc='none', ec='k')
pp1.loc2 = 4

ax21ins.xaxis.set_visible(False)

ax21ins.yaxis.set_label_position("right")
ax21ins.yaxis.tick_right()
ax21ins.yaxis.set_minor_locator(AutoMinorLocator(5))

ax21ins.text(0.05, 0.85, "noise floor", fontweight="bold", fontsize=8, transform=ax21ins.transAxes)

#%% set axis
ax00.set_xlim(wide)
ax01.set_xlim(narrow)

ax01.set_ylim([0.585, 0.5959])
ax11.set_ylim([0.18, 0.65])
ax21.set_ylim([0.31, 1.099])
ax00.set_ylim([0,1.1])
ax10.set_ylim([0,1.1])
ax20.set_ylim([0,1.1])


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


#%% red box to highlight zoomed region
alpha = 1
ax00.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)
ax10.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)
ax20.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)
ax01.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)
ax11.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)
ax21.axvspan(narrow[0], narrow[1], alpha=alpha, color=colors[-1], zorder=0)


#%% labels
ax21.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax20.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

ax00.set_ylabel('Intensity (arb.)')
ax10.set_ylabel('Intensity (arb.)')
ax20.set_ylabel('Transmission')


#%%

h = 0.02
v = 0.9

ax00.text(h, v, "A", fontweight="bold", fontsize=12, transform=ax00.transAxes)
ax01.text(h, v, "B", fontweight="bold", fontsize=12, transform=ax01.transAxes)
ax10.text(h, v, "C", fontweight="bold", fontsize=12, transform=ax10.transAxes)
ax11.text(h, v, "D", fontweight="bold", fontsize=12, transform=ax11.transAxes)
ax20.text(h, v, "E", fontweight="bold", fontsize=12, transform=ax20.transAxes)
ax21.text(h, v, "F", fontweight="bold", fontsize=12, transform=ax21.transAxes)


#%% save it

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\ISMS plots\mini.eps',bbox_inches='tight')


