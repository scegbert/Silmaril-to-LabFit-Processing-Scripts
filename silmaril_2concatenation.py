# -*- coding: utf-8 -*-
"""

silmaril2 - concatenation

Created on Tue Sep  1 16:18:21 2020

@author: scott

combine separate silmaril files as needed, look at the resultant noise, and save them as pckl'd files for easier access later

"""

from sys import path
path.append(r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\modules')

import pickle 
import os

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld
import td_support as td


forder = 3 # low pass filter for calculating residual noise
fcutoff = 0.02

snip = 165 # where you looked at noise in IGOR
start = 800 # range within the SNIP file to use
stop = 860

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\2021-09-03'
d_base = 'pre vac'
d_type = 'vac' # 'vac' or 'pure' or 'air'
d_vac_num = 29

d_ref = ' ref' # if you also saved a reference channel, such as to monitor background water, string for name, False if you didn't

num = 2
keep = [None, None, None] # which ones to keep from given file. None means keep them all 

# Locate phase-corrected DCS transmission spectra
# -----------------------------------------------

d_file = [' part 1', ' part 2', ' third', ' fourth', ' fifth', ' sixth', ' seventh'] # ['','','','','','']
weights = np.zeros(num)

plt.figure()

for i in range(num):
    
    print(d_file[i])
    
    file_SNIP = os.path.join(d_folder, d_base + d_file[i] + '_ch0_SNIP.txt')
    trans_snip = np.loadtxt(file_SNIP, skiprows=1)    

    file_AVG = os.path.join(d_folder, d_base + d_file[i] + '_ch0_AVG.txt')
    trans_avg = np.loadtxt(file_AVG, skiprows=7)
    
    if d_ref: 
        
        file_SNIP_ref = os.path.join(d_folder, d_base + d_file[i] + d_ref + '_ch0_SNIP.txt')
        trans_snip_ref = np.loadtxt(file_SNIP_ref, skiprows=1)

        file_AVG_ref = os.path.join(d_folder, d_base + d_file[i] + d_ref + '_ch0_AVG.txt')
        trans_avg_ref = np.loadtxt(file_AVG_ref, skiprows=7)
    
    # %% regretable addition due to measurements where baseline changed during same condition. sorry. 
    r'''
    if d_base == '300 K 1_5 T': 
    
        print('are you sure you want to do this? (if you are, remember you need to ignore the trans_snip now)')    
        
        d_load = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\new data\vacuum scans\BL filtered 1 0.010.pckl'
        
        f = open(d_load, 'rb')
        [bl_filt_all, wvn_bl] = pickle.load(f) # load in the smoothed laser baselines
        f.close()
    
        d_load = os.path.join(r'C:\Users\scott\Documents\1-WorkStuff\water campaign\new data\pure water', d_base + ' model.pckl')
        
        f = open(d_load, 'rb')
        [model_TD] = pickle.load(f)
        f.close()

        model_abs = np.real(np.fft.rfft(model_TD)) 
        model_trans = np.exp(-model_abs)
    
        spike_location = 11726 # we're shooting from the hip here. these are constant for this data set. only used for backtracking to process outliers
        istart = 31113 + spike_location
        istop = 177776 + spike_location
        
        trans_smoothed = trans_avg[istart:istop] / bl_filt_all[bl[i],:] # things are getting weird. This is a test for now
        spacing = np.arange(0,len(trans_smoothed))
        
        fit_order = 50
        fit_results = np.polynomial.chebyshev.Chebyshev.fit(spacing[model_trans > 0.9999], trans_smoothed[model_trans > 0.9999], fit_order)
    
        trans_smoothed2 = trans_smoothed / fit_results(spacing)
        
        trans_avg = trans_avg / max(trans_avg) # normalize the wings
        trans_avg[istart:istop] = trans_smoothed2 
        trans_avg = trans_avg * np.shape(trans_snip)[1] # scale by number of files
    r'''
    # %% back to our usual programming. again. sorry for the weirdness of that (unless you need it, then you're welcome)
     
    plt.plot(trans_avg / max(trans_avg), label=d_file[i])
    if d_ref: plt.plot(trans_avg_ref / max(trans_avg_ref), label=d_file[i] + ' - ref')
    
    with open(file_AVG,'r') as f:
        f.readline().split()[-1] # ignore the first line (it just says the name of the Igor wave we saved)
        
        coadds = int(f.readline().split()[-1])
    
        dfrep = float(f.readline().split()[-1])
        frep1 = float(f.readline().split()[-1]) 
        frep2 = float(f.readline().split()[-1])
    
        ppig = int(f.readline().split()[-1])
        fopt = float(f.readline().split()[-1])
 
    if keep: 
        if keep[i]:
            trans_snip = trans_snip[:,keep[i][0]-1:keep[i][1]-1]
            if d_ref: trans_snip_ref = trans_snip_ref[:,keep[i][0]-1:keep[i][1]-1]
            
    weights[i] = trans_snip.size / 1001 # the file is 1001 x number of coadds
    
    if i == 0:
        
        ALL_trans_avg = trans_avg
        ALL_trans_snip = trans_snip
        if d_ref: 
            ALL_trans_avg_ref = trans_avg_ref
            ALL_trans_snip_ref = trans_snip_ref
        
        ALL_coadds = coadds
        
        ALL_dfrep = dfrep
        ALL_frep1 = frep1
        ALL_frep2 = frep2
        
        ALL_ppig = ppig
        ALL_fopt = fopt
        
        num_scans = np.round(np.shape(trans_snip)[1],2)

    else: 
        
        ALL_trans_avg = np.column_stack((ALL_trans_avg,trans_avg))
        ALL_trans_snip = np.column_stack((ALL_trans_snip,trans_snip))
        if d_ref: 
            ALL_trans_avg_ref = np.column_stack((ALL_trans_avg_ref,trans_avg_ref))
            ALL_trans_snip_ref = np.column_stack((ALL_trans_snip_ref,trans_snip_ref))
        
        ALL_coadds = np.column_stack((ALL_coadds,coadds))
        
        ALL_dfrep = np.column_stack((ALL_dfrep,dfrep))
        ALL_frep1 = np.column_stack((ALL_frep1,frep1))
        ALL_frep2 = np.column_stack((ALL_frep2,frep2))
        
        ALL_ppig = np.column_stack((ALL_ppig,ppig))
        ALL_fopt = np.column_stack((ALL_fopt,fopt))
        
        num_scans = np.column_stack((num_scans,np.round(np.shape(trans_snip)[1],2)))
   
if i != 0: # if there was more than 1 file, average them together

    trans_avg = np.sum(ALL_trans_avg, axis=1) # each individual IG set is a sum (for now) - they're already weighted
    trans_snip = ALL_trans_snip
    if d_ref: 
        trans_avg_ref = np.sum(ALL_trans_avg_ref, axis=1) # each individual IG set is a sum (for now) - they're already weighted
        trans_snip_ref = ALL_trans_snip_ref
        

    coadds = np.average(ALL_coadds, axis=1, weights=weights)[0]
    
    dfrep = np.average(ALL_dfrep, axis=1, weights=weights)[0]
    frep1 = np.average(ALL_frep1, axis=1, weights=weights)[0]
    frep2 = np.average(ALL_frep2, axis=1, weights=weights)[0]
    
    ppig = np.average(ALL_ppig, axis=1, weights=weights)[0]
    fopt = np.average(ALL_fopt, axis=1, weights=weights)[0]
    
    plt.plot(trans_avg / max(trans_avg), label='final')
    if d_ref: plt.plot(trans_avg_ref / max(trans_avg_ref), label='final - ref')
    plt.legend(loc='upper right')

# verify that the noise looks good
# -----------------------------------------------

t_acq = coadds / dfrep / 60 # data collection time in minutes
    
b, a = signal.butter(forder, fcutoff)
trans_filt = signal.filtfilt(b, a, trans_snip, axis=0)

noise_sum = np.cumsum(trans_snip, axis=1) / np.cumsum(trans_filt, axis=1)
noise_std = np.std(noise_sum[start:stop,:],axis=0) # sometimes the edges are weird with the filtering routine

noise_sum_time = trans_snip / trans_filt
noise_std_time = np.std(noise_sum_time[start:stop,:],axis=0)

t_sum = t_acq * np.arange(1,np.shape(trans_snip)[1]+0.1,1)

m, c = np.polyfit(np.log(t_sum), np.log(noise_std), 1) # fit log(noise) = m*log(time) + c

t_fit = np.linspace(t_acq, max(t_sum))
noise_fit = np.exp(m*np.log(t_fit) + c) # calculate the fitted values of noise

r'''
plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.plot(trans_filt[:,0])
plt.plot(trans_filt[:,int(np.shape(trans_snip)[1]/2)])
plt.plot(trans_filt[:,-1])

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.plot(noise_sum[:,0], label='1')
plt.plot(noise_sum[:,-1], label='1->'+str(np.shape(trans_snip)[1]))
plt.legend()
r'''

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.loglog(t_sum, noise_std, 'xr')
plt.loglog(t_fit, noise_fit, 'k')
plt.xlabel('duration of data collection (minutes)')
plt.ylabel('noise')
plt.legend()
plt.grid(which='both')


plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.plot(t_sum, noise_std_time)
plt.xlabel('duration of data collection (minutes)')
plt.ylabel('noise during each scan (not averaged)')
plt.grid(which='both')

print('estimated final noise = ', np.round(noise_std[-1]*10000,2),'e-4') # with IGs excluded in python


if d_ref: # monitor the reference channel
    
    bottom = 319
    baseline = 260
    buffer = 2
    
    r'''
    k = -3
    plt.figure()
    plt.plot(ALL_trans_snip_ref[:,k])
    plt.plot(bottom, ALL_trans_snip_ref[bottom,k], 'x', )
    plt.plot(baseline, ALL_trans_snip_ref[baseline,k], 'x')
    r'''
    
    trans_baseline = sum(ALL_trans_snip_ref[baseline-buffer:baseline+buffer])
    trans_bottom = sum(ALL_trans_snip_ref[bottom-buffer:bottom+buffer])
    
    ref_conc = (trans_baseline - trans_bottom) / trans_baseline
    ref_avg = 100 * np.average(ref_conc)
    ref_std = 100 * np.std(ref_conc)
    
    r'''
    plt.figure()
    for j in [1,15,30]:
        
        trans_ratio = []
        time = []
        i = j
        while i < len(trans_baseline): 
            trans_ratio.append(100*sum(trans_baseline[i-j:i] - trans_bottom[i-j:i]) / sum(trans_baseline[i-j:i]))
            time.append(i-j/2)
            i +=1
        
        plt.plot(time, trans_ratio)
        plt.ylabel('percent transmission of background water')
        plt.xlabel('time (minutes)')
    r'''

trans_filt_avg = signal.filtfilt(b, a, trans_avg[snip*1000+start:snip*1000+stop], axis=0) # match up with snip (hopefully)
noise_sum_avg = trans_avg[snip*1000+start:snip*1000+stop] / trans_filt_avg
noise_std_avg = np.std(noise_sum_avg,axis=0)
print('final noise = ', np.round(noise_std_avg*10000,2),'e-4')
if d_ref: print('reference concentration = ', np.round(ref_avg,3), '+/-', np.round(ref_std,3))

if num == 1: print('duration of measurement = ', num_scans, ' scans')
else: print('duration of measurement = ', np.sum(num_scans[0]), ' scans, ', num_scans[0])

# save the completed file
# -----------------------------------------------


dfg = dfgdfg

# compile noise for data sets 
if d_type == 'pure':
    
    d_final = os.path.join(r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water', d_base + ' pre.pckl')

elif d_type == 'air':
    
    d_final = os.path.join(r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water', d_base + ' pre.pckl')

else:

    d_final = os.path.join(r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans', d_type + ' ' + str(d_vac_num) + ' pre.pckl')


f = open(d_final, 'wb')
if d_ref: pickle.dump([trans_avg, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, trans_avg_ref, trans_snip_ref], f)
else: pickle.dump([trans_avg, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt], f)
f.close() 



