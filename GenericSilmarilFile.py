

import os
import numpy as np
import matplotlib.pyplot as plt

from sys import path
path.append(r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\modules')
import pldspectrapy as pld
import td_support as td

pld.db_begin('data - HITRAN 2016')  # load the linelists into Python (keep out of the for loop)


type = 'air' # 'air' 'pure'
check_model = True

if type == 'air':

    d_folders = [r'C:\Users\scott\Downloads'] 
    d_files = ['300 K 600 T real'] # name of the file without _ch0_AVG.txt at the end.
    file_names = ['']
    P = 600
    
    spike_expected = [13979, 13979]
      
    P_vary = False
    yH2O = 0.01
    yH2O_vary = True
    
elif type == 'pure':

  
    d_folders = [r'C:\Users\scott\Documents\1-WorkStuff\water campaign\distortion check\change detector voltage'] 
    d_files =  ['300 K 16 T']  # name of the file without _ch0_AVG.txt at the end.
    file_names = ['']
    P = 16.0
    P_vary = True
    yH2O = 1
    yH2O_vary = True
    
    spike_expected = [13979]

data = {}

colors = ['navy', 'dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'limegreen', 'deeppink', 'purple']


nyq_side = 1
wvn2_range = [6600,7600]
wvn2_fit = [6900, 7000] 

pathlength = 91.4
T = 22 + 273
T_vary = False

bl_cutoff_fit = 50
etalons = []

meas_conditions = np.zeros((len(d_files),8)) # initialize variable to hold all results


# %% Locate phase-corrected DCS transmission spectra

for i in range(len(d_files)):
    
    d_file_avg = os.path.join(d_folders[i], d_files[i] + '_ch0_AVG.txt')
    
    meas_w_spike = np.loadtxt(d_file_avg, skiprows=7)
    
    with open(d_file_avg,'r') as f:
    	f.readline().split()[-1] # ignore the first line (it just says the name of the Igor wave we saved)
    	
    	coadds = int(f.readline().split()[-1])
    
    	dfrep = float(f.readline().split()[-1])
    	frep1 = float(f.readline().split()[-1]) 
    	frep2 = float(f.readline().split()[-1])
    
    	ppig = int(f.readline().split()[-1])
    	fopt = float(f.readline().split()[-1])
    	
    
    # %% Locate and remove spike
    
    spike_location = np.argmax(np.abs(np.diff(meas_w_spike, n=5))) + 2 # the nyquist window starts with a large delta function (location set by shift in Silmatil GUI)
    
    meas_raw = meas_w_spike[spike_location:-1]
    
    if spike_location != spike_expected[i]: # this number will be different for each measurement, check the location of the spike in your data if python isn't finding it
        print('***** manual entry of spike location *****')
        spike_location = spike_expected[i]
        meas_raw = meas_w_spike[spike_location:-1] # the noisy area at 45,000 is throwing off this algorithm. for the entire water data set this should do fine
        
    # %% Calculate DCS frequency axis
    
    hz2cm = 1 / 299792458 / 100 # conversion from MHz to cm-1 using speed of light
    
    fclocking = frep2 # really it was comb 1, but for some reason this is what silmaril data wants
    favg = np.mean([frep1,frep2])
    
    nyq_range = ppig*2 * fclocking
    nyq_num = np.round(1e6 / (ppig*2))
    
    fcw = nyq_range * nyq_num
    
    nyq_start = fcw + fopt
    
    if nyq_side == 1:
        nyq_stop = nyq_range * (nyq_num + 0.5) + fopt # Nyquist window 1.0 - 1.5
        wvn_raw = np.arange(nyq_start, nyq_stop, favg) * hz2cm
        wvn_raw = wvn_raw[:len(meas_raw)]
            
    elif nyq_side == -1:
        nyq_stop = (nyq_range * (nyq_num - 0.5) + fopt) # Nyquist window 0.5 - 1.0
        wvn_raw = np.arange(nyq_start, nyq_stop, -favg) * hz2cm
        wvn_raw = wvn_raw[:len(meas_raw)]
    
        meas_raw = np.flip(meas_raw)
        wvn_raw = np.flip(wvn_raw)

    # %% spectrally trim and normalize 
    
    istart = np.argmin(abs(wvn_raw - wvn2_range[0]))-1 # for prime factor
    istop = np.argmin(abs(wvn_raw - wvn2_range[1]))
    
    wvn = wvn_raw[istart:istop]
    meas = meas_raw[istart:istop] / max(meas_raw[istart:istop])
    meas_TD = np.fft.irfft(-np.log(meas))
    
    istart = np.argmin(abs(wvn_raw - wvn2_fit[0]))-3 # for prime factor
    istop = np.argmin(abs(wvn_raw - wvn2_fit[1]))
    
    wvn_fit = wvn_raw[istart:istop]
    meas_fit = meas_raw[istart:istop] / max(meas_raw)
   
    meas_fit_TD = np.fft.irfft(-np.log(meas_fit))
    
   # %% make a model to help baseline correct the measurement (not TD for simplicity)

    if check_model: 
        mod, pars = td.spectra_single_lmfit()
        
        pars['mol_id'].value = 1 # water = 1 (hitran molecular code)
        pars['pathlength'].set(value = pathlength, vary = False) # pathlength in cm
        
        pars['temperature'].set(value = T, vary = T_vary) # temperature in K
        pars['pressure'].set(value = P/760, vary = P_vary) # pressure in atm (converted from Torr)
        pars['molefraction'].set(value = yH2O, vary = yH2O_vary) # mole fraction
        
        model_fit_TD = mod.eval(xx=wvn_fit, params=pars, name='H2O')
        model = np.exp(-np.real(np.fft.rfft(model_fit_TD)))
        
        weight = td.weight_func(len(wvn_fit), bl_cutoff_fit, etalons)   
        
        model_lmfit = mod.fit(meas_fit_TD, xx=wvn_fit, params=pars, weights=weight)
        # td.plot_fit(wvn_fit, model_lmfit)
        
        yH2O_fit = model_lmfit.params['molefraction'].value
        yH2O_fit_unc = model_lmfit.params['molefraction'].stderr
        
        P_fit = model_lmfit.params['pressure'].value * 760
        try: P_fit_unc = model_lmfit.params['pressure'].stderr * 760
        except: P_fit_unc = model_lmfit.params['pressure'].stderr
        
        T_fit = model_lmfit.params['temperature'].value
        T_fit_unc = model_lmfit.params['temperature'].stderr
        
        shift_fit = model_lmfit.params['shift'].value
        shift_fit_unc = model_lmfit.params['shift'].stderr
        
        meas_conditions[i,:] = [yH2O_fit, yH2O_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc]
        
        pars['temperature'].set(value = T_fit, vary = True) # temperature in K
        pars['pressure'].set(value = P_fit/760, vary = P_vary) # pressure in atm (converted from Torr)
        pars['molefraction'].set(value = yH2O_fit, vary = yH2O_vary) # mole fraction
        
        model_TD = mod.eval(xx=wvn, params=pars, name='H2O')   
        model = np.exp(-np.real(np.fft.rfft(model_TD)))
        
        data[file_names[i]] = [wvn, meas, model]  # save for later

    else: 
        
        data[file_names[i]] = [wvn, meas]  # save for later

# %% baseline correction and data visualization

for i in range(len(d_files)):

    try: 
        meas_dif = meas - data[file_names[i]][1]    
        meas_ratio = meas / data[file_names[i]][1]    
    except: pass
        
    wvn = data[file_names[i]][0]
    meas = data[file_names[i]][1]
    
    if check_model: 
        model = data[file_names[i]][2]
        
        #% chebyshev baseline correction
        
        fit_order = 50 
        trans_cutoff = 0.985
        fit_results = np.polynomial.chebyshev.Chebyshev.fit(wvn[model > trans_cutoff], meas[model > trans_cutoff], fit_order) 
        bl = fit_results(wvn) # improved baseline correction for entire spectrum
           
        try: 
            meas_cheby_dif = meas_cheby - (meas / bl)
            meas_cheby_ratio = meas_cheby / (meas / bl)
        except: pass
        meas_cheby = meas / bl
    
        #% time domain baseline correction
        
        meas_TD = np.fft.irfft(-np.log(meas))
        model_TD = np.fft.irfft(-np.log(model))
        
        bl_cutoff_all = 30
        weight = td.weight_func(len(wvn), bl_cutoff_all, etalons)   
        
        meas_TDBL_TD = meas_TD*weight + model_TD*(1-weight)
        meas_TDBL = np.exp(-np.real(np.fft.rfft(meas_TDBL_TD)))
    
        #% plots
    
        plt.figure(1, constrained_layout=True)
        plt.plot(wvn, meas_cheby, color=colors[i], label=file_names[i])
        plt.plot(wvn, model, '--', color=colors[i])
        plt.plot(wvn, model-meas_cheby + 1.05, color=colors[i])
        plt.legend(loc='lower left')
        plt.title('Chebyshev baseline correction ('+str(fit_order)+'th order across 800 cm-1)')
        plt.xlabel('wavenumber')
        plt.ylabel('transmission')    
    
        plt.figure(2, constrained_layout=True)
        plt.plot(wvn, meas_TDBL, color=colors[i], label=file_names[i])
        plt.plot(wvn, model, '--', color=colors[i])
        plt.plot(wvn, model-meas_TDBL+1.05, color=colors[i])
        plt.legend(loc='lower left')
        plt.title('Time Domain') 
        plt.xlabel('wavenumber')
        plt.ylabel('transmission')    
        
            
        plt.figure(4, constrained_layout=True)
        plt.plot(wvn,meas_cheby, color=colors[i], label=file_names[i])
        plt.legend(loc='lower left')
        
    else:         
        
        plt.figure(3, constrained_layout=True)
        plt.plot(wvn,meas,label=file_names[i])     
      
    plt.figure(5, constrained_layout=True)
    plt.plot(wvn, meas, label=file_names[i])

    if i % 2 != 0: 
        
        if check_model: 
            
            plt.figure(4, constrained_layout=True)
            plt.plot(wvn,meas_cheby_dif, color=colors[i], label='('+file_names[i-1] +'-'+ file_names[i]+')')
            plt.legend(loc='lower left')
            plt.title('measurements only ('+str(fit_order)+'th order across 800 cm-1)') 
            plt.xlabel('wavenumber')
            plt.ylabel('transmission')    
        

            plt.figure(4*2, constrained_layout=True)
            plt.plot(wvn,meas_cheby_dif, color=colors[i], label='('+file_names[i-1] +'-'+ file_names[i]+')')
            plt.legend(loc='lower left')
            plt.title('measurements only ('+str(fit_order)+'th order across 800 cm-1)') 
            plt.xlabel('wavenumber')
            plt.ylabel('transmission')    


        else: 
                
            plt.figure(3, constrained_layout=True)
            plt.plot(wvn,meas_dif, color=colors[i], label='('+file_names[i-1] +'-'+ file_names[i]+')')
            plt.legend(loc='lower left')
        
            plt.figure(3*2, constrained_layout=True)
            plt.plot(wvn,meas_dif, color=colors[i], label='('+file_names[i-1] +'-'+ file_names[i]+')')
            plt.legend(loc='lower left')

        
        plt.figure(5, constrained_layout=True)
        plt.plot(wvn, meas_ratio, color=colors[i], label='('+file_names[i-1] +'/'+ file_names[i]+')')    
        plt.legend(loc='lower left')
    
        plt.figure(5*2, constrained_layout=True)
        plt.plot(wvn, meas_ratio, color=colors[i], label='('+file_names[i-1] +'/'+ file_names[i]+')')    
        plt.legend(loc='lower left')











