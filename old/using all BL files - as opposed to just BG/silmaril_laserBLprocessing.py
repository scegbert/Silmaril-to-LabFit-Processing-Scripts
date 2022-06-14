"""
Created on Thu Aug 13 08:42:46 2020

@author: scott
"""

from scipy import signal
from scipy import interpolate
import numpy as np
import os
import pickle 
import matplotlib.pyplot as plt

from packfind import find_package
find_package('pldspectrapy')
import pldspectrapy as pld
import td_support as td


# %% dataset specific information

nyq_side = 1 # which side of the Nyquist window are you on? + (0 to 0.25) or - (0.75 to 0)

wvn2_range = [6600,7600] 
wvn2_fit = [7000,7300] # area with strong water and smoother baseline

wvn_buffer = 30 # cm-1 buffer to smooth edges of the filter

ncutoff = 0.9995
nbuffer = 30 # median filter for features

forderLP = 1
fcutoffLP = '0.030'

d_filter = str(forderLP) + ' ' + fcutoffLP
fcutoffLP = float(fcutoffLP)

pathlength = 2*167.6 - 91.4 + 15 # cm, furnace - cell + 4f
P_guess = 0.8 # atm
T_guess = 300 # K
yh2o_guess = 1e-5 # percent water in air

td_cutoff = 80

d_fit = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\Pure Water Data\final\BL fit.pckl'

# %% Locate phase-corrected measured spectrum

d_final = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\Pure Water Data\final\BL final.pckl'
f = open(d_final, 'rb')
BL_all = pickle.load(f)
f.close() 

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\Pure Water Data\final' # grab one of the files for generic frequency values
d_base = '300 K _5 T'
d_ref = os.path.join(d_folder, d_base + ' final.pckl')
f = open(d_ref, 'rb')
[trans_avg, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt] = pickle.load(f)
f.close() 
    
# %% Locate and remove spike

spike_location = np.argmax(np.abs(np.diff(trans_avg, n=5))) + 3 # the nyquist window starts with a large delta function (location set by shift in Silmatil GUI)
meas_ref = trans_avg[spike_location:-1]

BL_raw = BL_all[spike_location:-1,:]

# %% Calculate DCS frequency axis

hz2cm = 1 / pld.SPEED_OF_LIGHT / 100 # conversion from MHz to cm-1

fclocking = frep2 # really it was comb 1, but for some reason this is what silmaril data wants
favg = np.mean([frep1,frep2])

nyq_range = ppig*2 * fclocking
nyq_num = np.round(1e6 / (ppig*2))

fcw = nyq_range * nyq_num

nyq_start = fcw + fopt

if nyq_side == 1:
    nyq_stop = nyq_range * (nyq_num + 0.5) + fopt # Nyquist window 1.0 - 1.5
    wvn_raw = np.arange(nyq_start, nyq_stop, favg) * hz2cm
    wvn_raw = wvn_raw[:len(meas_ref)]
        
elif nyq_side == -1:
    nyq_stop = (nyq_range * (nyq_num - 0.5) + fopt) # Nyquist window 0.5 - 1.0
    wvn_raw = np.arange(nyq_start, nyq_stop, -favg) * hz2cm
    wvn_raw = wvn_raw[:len(meas_ref)]

    meas_ref = np.flip(meas_ref)
    wvn_raw = np.flip(wvn_raw)
    BL_raw = np.flipud(BL_raw)


# %% spectrally trim the measurement

ibuffer = int(wvn_buffer / (wvn_raw[2] - wvn_raw[1])) # convert wavenumber buffer to number of points

istart = np.argmin(abs(wvn_raw - wvn2_range[0]))
istop = np.argmin(abs(wvn_raw - wvn2_range[1]))

wvn = wvn_raw[istart-ibuffer:istop+ibuffer]
BL_prefilt = BL_raw[istart-ibuffer:istop+ibuffer,:] / np.max(BL_raw[istart-ibuffer:istop+ibuffer,:],axis=0)
BL_prefilt_TD_fit = np.fft.irfft(-np.log(BL_prefilt), axis=0)

# %% fit the data (if it hasn't already been done)

try: # hopefully you have already fit for the water spectrum that needs to be removed
    
    f = open(d_fit, 'rb')
    [model_TD_fit, output] = pickle.load(f)
    f.close()
    
except: # if you haven't, there's no time like the present
    
    istart = np.argmin(abs(wvn - wvn2_fit[0]))
    istop = np.argmin(abs(wvn - wvn2_fit[1]))
    
    wvn_fit = wvn[istart:istop]
    BL_fit = BL_prefilt[istart:istop,:] / np.max(BL_prefilt[istart:istop,:], axis=0)
    
    BL_TD_fit = np.fft.irfft(-np.log(BL_fit), axis=0)
    
    mod, pars = td.spectra_single_lmfit()
    
    pars['mol_id'].value = 1 # water = 1 (hitran molecular code)
    pars['pathlength'].set(value = pathlength, vary = False) # pathlength in cm
    
    pars['molefraction'].set(value = yh2o_guess, vary = True) # mole fraction
    pars['pressure'].set(value = P_guess, vary = True) # pressure in atm 
    pars['temperature'].set(value = T_guess, vary = True) # temperature in K
    
    pld.db_begin('data - HITRAN')  # load the linelists into Python
    model_TD_fit = mod.eval(xx=wvn_fit, params=pars, name='H2O')
    
    residual = BL_TD_fit - model_TD_fit[:,None]
    
    plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(residual)
    plt.ylabel('time domain response')
    plt.legend()
    
    weight = td.weight_func(len(wvn_fit), td_cutoff, etalons = [])
    
    output = np.zeros((8,len(BL_TD_fit[0,:])))
    model_TD_fit = np.zeros_like(BL_prefilt_TD_fit)
    
    for i in range(len(BL_TD_fit[0,:])):
        
        print(i)
        
        meas_fit_BG = mod.fit(BL_TD_fit[:,i], xx=wvn_fit, params=pars, weights=weight)
        td.plot_fit(wvn_fit, meas_fit_BG)
        
        yh2o_fit = meas_fit_BG.params['molefraction'].value 
        yh2o_fit_unc = meas_fit_BG.params['molefraction'].stderr 
        
        P_fit = meas_fit_BG.params['pressure'].value 
        P_fit_unc = meas_fit_BG.params['pressure'].stderr 
        
        T_fit = meas_fit_BG.params['temperature'].value
        T_fit_unc = meas_fit_BG.params['temperature'].stderr
        
        shift_fit = meas_fit_BG.params['shift'].value
        shift_fit_unc = meas_fit_BG.params['shift'].stderr
        
        output[:,i] = [yh2o_fit, yh2o_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc]
    
        pars['molefraction'].set(value = yh2o_fit, vary = True) # mole fraction
        pars['pressure'].set(value = P_fit, vary = True) # pressure in atm 
        pars['temperature'].set(value = T_fit, vary = True) # temperature in K
        pars['shift'].set(value = shift_fit, vary = True) # shift (cm-1)
        
        model_TD_fit[:,i] = mod.eval(xx=wvn, params=pars, name='H2O')    
        
    d_fit = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\Pure Water Data\final\BL fit.pckl'
    
    f = open(d_fit, 'wb')
    pickle.dump([model_TD_fit, output], f)
    f.close()

# %% remove absorption features from the measured spectra

BL_fit_TD = BL_prefilt_TD_fit - model_TD_fit

BL_fit = np.exp(-np.real(np.fft.rfft(BL_fit_TD, axis=0)))


# %% filter the data and remove the buffer

b, a = signal.butter(forderLP, fcutoffLP)
BL_filt = signal.filtfilt(b, a, BL_fit, axis=0)

wvn_ = wvn[ibuffer:-ibuffer]
BL_prefilt_ = BL_prefilt[ibuffer:-ibuffer]
BL_fit_ = BL_fit[ibuffer:-ibuffer]
BL_filt_ = BL_filt[ibuffer:-ibuffer]


# %% look at the data

iplot = 1

plt.figure()
plt.plot(wvn_,BL_prefilt_[:,iplot], label='raw')
plt.plot(wvn_,BL_fit_[:,iplot], label='TD water removal')
plt.plot(wvn_,BL_filt_[:,iplot], label='lowpass')

plt.legend()
plt.xlabel('wavenumber (cm-1)')
plt.ylabel('normalized laser intensity')

# %% save the data

please = stop

d_final = os.path.join(r'C:\Users\scott\Documents\1-WorkStuff\water campaign\Pure Water Data\final\BL ' + d_filter +'.pckl')

f = open(d_final, 'wb')
pickle.dump([BL_filt_, BL_all], f)
f.close() 





