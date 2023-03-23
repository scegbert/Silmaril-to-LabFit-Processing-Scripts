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
# d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - sceg'

# f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
# [T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure] = pickle.load(f)
# f.close()

# transmission_labfit = trans_pure

#%% load in transmission data (vacuum normalized from bg subtract)

d_measurement = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'
# f = open(os.path.join(d_measurement, which_file+' bg subtraction.pckl'), 'rb')
# # [meas_trans_bg, meas_trans_bl, wvn, T, P, y_h2o, pathlength, favg, fitresults_all, model_trans_fit2020, model_trans_fit2016, model_trans_fitPaul]
# [transmission, _, wvn_process, _, _, _, _, _, _, model, _, _] = pickle.load(f)
# f.close()

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

i_skip = 2293 # skip negative frequencies (and ditch noise spike)

measurement = measurement[i_skip:] / max(measurement[i_skip:])
wvn_raw = wvn_raw[i_skip:]
        


#%% load in vacuum scans

# d_vacuum = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'

# f = open(os.path.join(d_vacuum, 'plot stuff 1 0.030 with 2Ts.pckl'), 'rb')
# # [meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final]
# [meas_filt_all_final, meas_filt_all_final_ref, meas_spike_all_final, meas_bg_all_final, meas_raw_all_final, wvn_all_final] = pickle.load(f)
# f.close() 

# vacuum_raw = meas_raw_all_final[which_vacuum]
# vacuum_h2o = meas_bg_all_final[which_vacuum]
# vacuum_smooth = meas_filt_all_final[which_vacuum]


#%% plot and verify this is what you want

# plt.plot(wvn_process, transmission) # verify this is the data you want
# plt.plot(wvn_process, model) # verify this is the data you want
# plt.plot(wvn_raw, measurement) # verify this is the data you want


# [istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_data, max_prime_factor=500, print_value=False)
# meas_data = measurement[istart:istop] / max(measurement[istart:istop])

# [istart, istop] = td.bandwidth_select_td(wvn_process, wvn2_data, max_prime_factor=500, print_value=False)
# vac_raw_data = vacuum_raw[istart:istop] / max(vacuum_smooth[istart:istop])
# vac_h2o_data = vacuum_h2o[istart:istop] / max(vacuum_smooth[istart:istop])
# vac_smooth_data = vacuum_smooth[istart:istop] / max(vacuum_smooth[istart:istop])

# trans_data = meas_data / vac_smooth_data


# wvn_data = wvn_process[istart:istop]


# plt.plot(wvn_data, trans_data) # verify this is the data you want
# plt.plot(wvn_data, vac_raw_data) # verify this is the data you want
# plt.plot(wvn_data, vac_h2o_data) # verify this is the data you want
# plt.plot(wvn_data, vac_smooth_data) # verify this is the data you want
# plt.plot(wvn_data, meas_data) # verify this is the data you want

# asdfasdfsd

#%%

fig = plt.figure(figsize=(10, 4))

# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=6
colors = ['#d95f02','#1b9e77','k','#7570b3','#66a61e','#e6ab02']

plt.plot(wvn_raw, measurement, label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
plt.ylabel('Intensity (arb.)')
plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')

# plt.legend(loc = 'lower center', framealpha=1, edgecolor='black', fontsize=9)

#%% scan through various widths

# xylims = {1: [7027.87, 7028.51, 0.82, 1.01], 
#           2: [7027.65, 7028.97, 0.82, 1.01], 
#           3: [7026.14, 7029.86, 0.66, 1.01], 
#           4: [7022.40, 7033.40, 0.48, 1.02], 
#           5: [7011.50, 7043.08, 0.40, 1.03], 
#           6: [6992.20, 7055.60, 0.35, 1.05]} #, 
#           # 7: [6695.11, 7082.92, 0.30, 1.1], 
#           # 8: [6899.79, 7143.48, 0.25, 1.1]} 
#           # 6: [6992.20, 7055.60, 0.35, 1.01], 
#           # 6: [6992.20, 7055.60, 0.35, 1.01], 
#           # 6: [6992.20, 7055.60, 0.35, 1.01], 
#           # 6: [6992.20, 7055.60, 0.35, 1.01], 
#           # 6: [6992.20, 7055.60, 0.35, 1.01], 
#           # 6: [6992.20, 7055.60, 0.35, 1.01]}
          
          
xylims_start = [7027.87, 7028.51,  0.82, 1.01]
xylims_stop =  [6390.00, 7896.48, -0.02, 1.04]

num = 10
          
for i in np.arange(0, num, 1): 
    
    step_xlow  = (xylims_start[0] - xylims_stop[0]) / num
    step_xhigh = (xylims_start[1] - xylims_stop[1]) / num
    
    step_ylow  = (xylims_start[2] - xylims_stop[2]) / num
    step_yhigh = (xylims_start[3] - xylims_stop[3]) / num
    
    fig = plt.figure(figsize=(10, 4))

    plt.plot(wvn_raw, measurement, label='100% $\mathregular{H_2O}$ at 1300 K 16 T')
    plt.ylabel('Intensity (arb.)')
    plt.xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
    
    plt.xlim([xylims_start[0]+i*step_xlow, xylims_start[1]+i*step_xhigh])
    # plt.ylim([xylims_start[2]+i*step_xlow, xylims_start[3]+i*step_xhigh])

    # plt.savefig(r'C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\plots\7-3big {}.png'.format(i),bbox_inches='tight')



