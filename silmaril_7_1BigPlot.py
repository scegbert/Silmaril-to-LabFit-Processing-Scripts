
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


#%%


# load in transmission data (measured and model)

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water\1300 K 16 T bg subtraction.pckl'
f = open(d_folder, 'rb')
# [meas_trans_bg, meas_trans_bl, wvn, T, P, y_h2o, pathlength, favg, fitresults_all, model_trans_fit2020, model_trans_fit2016, model_trans_fitPaul]
[transmission, _, wvn, _, _, _, _, _, _, model, _, _] = pickle.load(f)
f.close()


# load in the raw measurement data

d_ref = True # there was a reference channel

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water\1300 K 16 T pre.pckl'
f = open(d_folder, 'rb')
# [measurement_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, trans_raw_ref, trans_snip_ref]
if d_ref: [measurement_w_spike, _, _, _, _, _, _, _, _, _] = pickle.load(f)
else: [measurement_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt] = pickle.load(f)
f.close()

spike_location_expected = 13979
measurement = measurement_w_spike[spike_location_expected:-1] / max(measurement_w_spike[spike_location_expected:-1]) # normalize max to 1

# load in the vacuum scan data

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'

d_load = os.path.join(d_folder, 'BL filtered 1 0.030 with 2Ts.pckl')
f = open(d_load, 'rb')
# [bl_filt_all, bl_filt_all_ref, wvn_bl]
if d_ref: [bl_filt_all, bl_filt_all_ref, wvn_bl] = pickle.load(f)
else: [bl_filt_all, wvn_bl] = pickle.load(f)
f.close()

vacuum_raw
vacuum_h2o
vacuum_smooth


plt.plot(wvn, transmission) # verify this is the data you want
plt.plot(wvn, model) # verify this is the data you want
plt.plot(wvn, measurement) # verify this is the data you want





#%%

fig = plt.figure(figsize=(10, 7.5))

gs = GridSpec(3, 2, width_ratios=[2, 1], hspace=0.015, wspace=0.005) # rows, columns

wide = [6601, 7599]
narrow = [7094.249, 7096.01] # [7093.9, 7096.1] # [7070.62, 7071.78] # [7089.5, 7100.5]
inset21 = [7094.81, 7095, 0.9985, 1.0019]

which_norm = (wavenumbers>6656) & (wavenumbers<7540)

colors = ['dodgerblue', 'firebrick', 'darkorange', 'dodgerblue', 'firebrick', 'moccasin']

ax00 = fig.add_subplot(gs[0,0]) # First row, first column
ax00.axvline(narrow[0], linewidth=1, color=colors[2])
ax00.axvline(narrow[1], linewidth=1, color=colors[2])
ax00.plot(wvnb,meas_rawb, color=colors[0], label='Laser Baseline (Gas Cell at Vacuum)')
ax00.plot(wvnb,meas_bgb, color=colors[1], label='Baseline without Background $\mathregular{H_2O}$')
ax00.plot(wvnb,meas_filtb, color=colors[2], label='Baseline after Low-pass Filter')
ax00.legend(loc = 'upper right', framealpha=1, edgecolor='black')

ax01 = fig.add_subplot(gs[0,1]) # First row, second column
ax01.plot(wvnb,meas_rawb, color=colors[0])
ax01.plot(wvnb,meas_bgb, color=colors[1])
ax01.plot(wvnb,meas_filtb, color=colors[2], linewidth=2)


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


