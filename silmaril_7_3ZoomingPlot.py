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
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

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

[T_all, P_all] = np.asarray([T_pure, P_pure])

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


hz2cm = 1 / pld.SPEED_OF_LIGHT / 100 # conversion from Hz to cm-1

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

i_skip = 2293 # skip negative frequencies (and ditch noise spike)

measurement = measurement[i_skip:] / max(measurement[i_skip:])
wvn_raw = wvn_raw[i_skip:]
        


#%% load in vacuum scans

d_vacuum = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'

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
trans_labfit = trans_labfit_all[istart:istop] 
res_updated = res_updated_all[istart:istop] 
res_og = res_og_all[istart:istop] 


sdfsdfsdf # need to fix wvn_labfit 

# plt.plot(wvn_data, trans_data) # verify this is the data you want
# plt.plot(wvn_data, vac_raw_data) # verify this is the data you want
# plt.plot(wvn_data, vac_h2o_data) # verify this is the data you want
# plt.plot(wvn_data, vac_smooth_data) # verify this is the data you want
# plt.plot(wvn_data, meas_data) # verify this is the data you want

# asdfasdfsd

#%% plot and scan through various widths

# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=6
# (#fee9ac is #e6ab02 lightened, same for #e7298a -> #f5a9d0, #66a61e was darkened to #4c7c17, same for #7570b3 to #514c8e)
colors = ['#d95f02','#1b9e77','k','#514c8e','#e7298a', '#4c7c17']

xylims_start = [7027.87, 7028.51,  0.82, 1.01]
xylims_stop =  [6390.00, 7896.48, -0.02, 1.04]

num_plots = 500
log_min = 1
log_max = 3

step_log = np.logspace(log_min,log_max,num_plots)
step_lin = np.linspace(log_min,log_max,num_plots)

fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_raw, measurement, color=colors[3], label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

for i in np.arange(0, num_plots, 1): 
    
    print(i)
    
    step_xlow  = (xylims_stop[0]-xylims_start[0]) / (step_log[-1]-step_log[0]) * (step_log[i]-step_log[0]) + xylims_start[0]
    step_xhigh = (xylims_stop[1]-xylims_start[1]) / (step_log[-1]-step_log[0]) * (step_log[i]-step_log[0]) + xylims_start[1]
    step_ylow  = (xylims_stop[2]-xylims_start[2]) / (step_lin[-1]-step_lin[0]) * (step_lin[i]-step_lin[0]) + xylims_start[2]
    step_yhigh = (xylims_stop[3]-xylims_start[3]) / (step_lin[-1]-step_lin[0]) * (step_lin[i]-step_lin[0]) + xylims_start[3]
       
    plt.xlim([step_xlow, step_xhigh])
    plt.ylim([step_ylow, step_yhigh])

    plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 zooming 10{}.png'.format(1000-i),bbox_inches='tight')


# plot the same stuff as above but showing the comb teeth (only for when we're zoomed in)

width_teeth = 10000 * hz2cm # 10 kHz to MHz to cm-1
extra_points_per = 10 # how many extra points below the width the comb tooth (one comb tooth is extra_points_per wide)
extra_points = 200000 # int((wvn_raw[1] - wvn_raw[0]) / width_teeth * extra_points_per) # replace each point with how many points? 1 -> extra_points

[istart, istop] = td.bandwidth_select_td(wvn_raw, xylims_start[0:2], max_prime_factor=500, print_value=False)

wvn_teeth_range = wvn_raw[istart:istop]
measurement_teeth_range = measurement[istart:istop]

wvn_teeth = interp1d(np.arange(len(wvn_teeth_range)), wvn_teeth_range, kind='linear')(
    np.linspace(0, len(wvn_teeth_range)-1, len(wvn_teeth_range) * extra_points - extra_points+1)) # expanded wavelength variable

measurement_teeth = np.zeros(len(wvn_teeth_range) * extra_points - extra_points+1)
measurement_teeth[::extra_points] = measurement_teeth_range

# Create a Gaussian kernel
measurement_teeth_conv = gaussian_filter1d(measurement_teeth, sigma=width_teeth*9000000, mode='constant')

# Normalize
measurement_teeth_conv = measurement_teeth_conv * max(measurement_teeth) / max(measurement_teeth_conv)

#%%

xylims_start_teeth = [7028.164532-1E-10, 7028.164532+1E-10,  -0.03000001, 1.01]
xylims_stop_teeth = [7027.87, 7028.51,  -0.03, 1.07]

num_plots_teeth = 300

step_log = np.logspace(log_min,log_max,num_plots_teeth+1)
step_lin = np.logspace(log_min,log_max,num_plots_teeth+1)

# for i in np.arange(0, num_plots_teeth, 1): 

fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_teeth, measurement_teeth_conv, color=colors[3], label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
    
for i in np.arange(0, num_plots_teeth, 1): 

    i+=1    
    
    print(i)
    
    step_xlow  = (xylims_stop_teeth[0]-xylims_start_teeth[0]) / (step_log[-1]-step_log[0]) * (step_log[i]-step_log[0]) + xylims_start_teeth[0]
    step_xhigh = (xylims_stop_teeth[1]-xylims_start_teeth[1]) / (step_log[-1]-step_log[0]) * (step_log[i]-step_log[0]) + xylims_start_teeth[1]
    step_ylow  = (xylims_stop_teeth[2]-xylims_start_teeth[2]) / (step_lin[-1]-step_lin[0]) * (step_lin[i]-step_lin[0]) + xylims_start_teeth[2]
    step_yhigh = (xylims_stop_teeth[3]-xylims_start_teeth[3]) / (step_lin[-1]-step_lin[0]) * (step_lin[i]-step_lin[0]) + xylims_start_teeth[3]
       
    plt.xlim([step_xlow, step_xhigh])
    plt.ylim([step_ylow, step_yhigh])
    
    plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 zooming teeth {}.png'.format(i-1),bbox_inches='tight')

    # if i == 2: asdfasdfsd

plt.plot(wvn_raw, measurement, color='k', label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 both zooming 1 black.png'.format(i),bbox_inches='tight')


fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_raw, measurement, color=colors[3], label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

plt.xlim([step_xlow, step_xhigh])
plt.ylim([step_ylow, step_yhigh])

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 both zooming 2.png'.format(i),bbox_inches='tight')

#%% plot various panels from big plot in a larger size for presentations

wide = [6615-50, 7650+25]

# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=6
# (#fee9ac is #e6ab02 lightened, same for #e7298a -> #f5a9d0, #66a61e was darkened to #4c7c17, same for #7570b3 to #514c8e)
colors = ['#d95f02','#1b9e77','k','#514c8e','#e7298a', '#4c7c17']
# colors = ['#d95f02','#1b9e77','k','#7570b3','#66a61e']


fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_raw, measurement, color=colors[3], label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
plt.xlim(wide)
plt.ylim([0,1.1])
plt.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=9)

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 big measurement.svg',bbox_inches='tight')


fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_data,vac_raw_data, color=colors[0], label='Baseline - Optical Cell at 1300 K <1mT')
plt.plot(wvn_data,vac_h2o_data, color=colors[1], label='Baseline - Background $\mathregular{H_2O}$ Removed')
plt.plot(wvn_data,vac_smooth_data, color=colors[2], label='Baseline - Low-pass Filtered')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
plt.xlim(wide)
plt.ylim([0,1.1])
plt.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=9)

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 big vacuum.svg',bbox_inches='tight')

# add an inset to zoom in on the filtering and background removal (step things in?)




fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_data, trans_data, color=colors[4], label='100% $\mathregular{H_2O}$ at 1300 K 16 T - Normalized by Filtered Baseline')
plt.ylabel('Transmission (partially normalized)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
plt.xlim(wide)
plt.ylim([0,1.1])
plt.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=9)

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 big normalized.svg',bbox_inches='tight')



fig = plt.figure(figsize=(10, 4))
plt.plot(wvn_labfit, trans_labfit/100, color=colors[5], label='100% $\mathregular{H_2O}$ at 1300 K 16 T - Normalized by Filtered Baseline and Chebyshev Polynomials')
plt.ylabel('Transmission')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
plt.xlim(wide)
plt.ylim([0,1.1])
plt.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=9)

plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3 big labfit.svg',bbox_inches='tight')














