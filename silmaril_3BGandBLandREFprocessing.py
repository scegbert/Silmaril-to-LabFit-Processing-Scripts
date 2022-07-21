"""
Created on Thu Sept 16 2020

@author: scott

"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

pld.db_begin('data - HITRAN 2020')  # load the linelists into Python (keep out of the for loop)

#%% dataset specific information
d_ref = True # did you use a reference channel? True or False 
include_meas_refs = True # include the reference channels for other measurements as background scans?  

two_background_temps = True # fit two background temperatures? (one for 4f, one for furnace)

calc_fits = False # fit the background data(or use previously fit data)
calc_background = True # generate the model for the fits (for background water subtraction) - False = load model (hopefully you have one saved)
remove_spikes = True # remove digital noise spikes from background scan before filtering 
save_results = True # save the things you calculate here? 

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'
d_file = 'vac '

spike_location_expected = 13979

forderLP = 1
fcutoffLP = '0.030'

if two_background_temps: 
    d_conditions = os.path.join(d_folder, 'BL conditions with 2Ts.pckl')
    d_bgcorrected = os.path.join(d_folder, 'BL BG corrected with 2Ts.pckl')
    d_filter = str(forderLP) + ' ' + fcutoffLP + ' with 2Ts'
else: 
    d_conditions = os.path.join(d_folder, 'BL conditions.pckl')
    d_bgcorrected = os.path.join(d_folder, 'BL BG corrected.pckl')
    d_filter = str(forderLP) + ' ' + fcutoffLP

fcutoffLP = float(fcutoffLP)

bl_number = 30 # there are 30 of them (0-29)
if include_meas_refs: 
    meas_pure_number = 29 # how many pure water files 
    meas_air_number = 29 # how many air water files
    bl_vac_number = bl_number
    bl_number += meas_pure_number + meas_air_number
    
    d_file_pure = ['300 K _5 T', '300 K 1 T',  '300 K 1_5 T','300 K 2 T',  '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
                   '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
                   '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
                   '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
                   '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T', 
                   '1300 K 16 T']
    d_folder_pure = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'

    d_file_air = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
                  '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
                  '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
                  '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
                  '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T', 
                  '1300 K 600 T']
    d_folder_air = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water'

nyq_side = 1 # which side of the Nyquist window are you on? + (0 to 0.25) or - (0.75 to 0)

wvn2_range_BL = [6500, 7800] # entire range we will want for the measurements
wvn_buffer = 30 # wavenumber buffer to avoid edge effects
wvn2_range_fit = [7050,7450] # [7000,7185] # focusing on strongest water features we can see in BG scans

vary_pressure = True # fit pressure? (advise yes, you can correct P*y back to y@Patm later if you'd like a stable value for y)
Patm = 628 # Torr

Tatm = 293
Tfurnace = [300, 300, 300, 300, 300, 300, 500, 500, 500, 500, 500, 500, 
            600, 700, 700, 900, 900, 900, 1100, 1100, 1100, 1100, 1100, 
            1100, 1100, 1200, 1200, 1200, 1200, 1200, 700]

pathlength_furnace = 2*160 - 91.4 # furnace (x2) - cell (double pass)
pathlength_4f = 15. # 4f (combs passed separately, but with nominally equal path length), this is the pathlength of the reference channel

yH2Obg = 2.5e-5 # guess at BG water concentration

bl_start = 150  # time domain baseline cutoff (etalon at 130)
bl_etalons = [[[745,765], [1477,1486], [48040,48100]], 
              [[750,760]], 
              [[750,760], [1477,1486]]] #updated for august 2021 data, including reference stuff (all the same) 

# 4 only 750 etalon
# 0-3 add 5 to each side of 750 (out of phase), new etalon way out at [48040,48100]

spike_threshold = 4 # how many standard deviations from average is a noise spike? 
spike_points_num = 30 # how many points on each side to remove

if two_background_temps: bl_conditions = np.zeros((bl_number,9*2)) # initialize variable to hold all results (2x for 2T's)
else: bl_conditions = np.zeros((bl_number,9)) # initialize variable to hold all results

if d_ref: bl_conditions_ref = np.zeros((bl_number,9)) # initialize variable to hold all results
spike_location = np.zeros(bl_number, dtype=int)

d_file_meas_all = []
meas_spike_all = np.zeros((bl_number,199434))
meas_bg_all = np.zeros((bl_number,199434))
meas_raw_all = np.zeros((bl_number,199434))

if d_ref: 
    meas_spike_all_ref = np.zeros((bl_number,199434))
    meas_bg_all_ref = np.zeros((bl_number,199434))
    meas_raw_all_ref = np.zeros((bl_number,199434))
wvn_all = np.zeros((bl_number,199434))
index_all_final = np.zeros((bl_number,2))

#%% Locate phase-corrected measured spectrum


for bl in range(bl_number):    
    
    if not include_meas_refs or bl < bl_vac_number: 
        print('******* loading vacuum scan ' + str(bl))
        d_file_meas = d_file +' '+ str(bl)
        d_final = os.path.join(d_folder, d_file + str(bl) + ' pre.pckl')
        
    elif bl >= bl_vac_number and bl_number-bl > meas_pure_number:
        print('******* loading pure water reference scan ' + str(bl) + ' (' + d_file_pure[bl-bl_vac_number] + ')')
        d_file_meas = d_file_pure[bl-bl_vac_number]
        d_final = os.path.join(d_folder_pure, d_file_meas + ' pre.pckl')

    else: 
        print('******* loading air-water reference scan ' + str(bl) + ' (' + d_file_air[bl-bl_vac_number-meas_pure_number] + ')')
        d_file_meas = d_file_air[bl-bl_vac_number-meas_pure_number]
        d_final = os.path.join(d_folder_air, d_file_meas + ' pre.pckl')
    
    d_file_meas_all.append(d_file_meas)
    
    f = open(d_final, 'rb')
    if d_ref: 
        [trans_raw_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, trans_raw_ref, trans_snip_ref] = pickle.load(f)
    else: 
        # [trans_raw_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt] = pickle.load(f) # if there isn't a reference channel
        [trans_raw_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, _, _] = pickle.load(f) # if there is a reference channel, but you don't want to use it right now
            
    f.close() 
    
    spike_location[bl] = np.argmax(np.abs(np.diff(trans_raw_w_spike, n=5))) + 2 # the nyquist window starts with a large delta function (location set by shift in Silmatil GUI)
    
    meas_raw = trans_raw_w_spike[spike_location[bl]:-1] / max(trans_raw_w_spike[spike_location[bl]:-1]) # normalize max to 1
    if d_ref: meas_raw_ref = trans_raw_ref[spike_location[bl]:-1] / max(trans_raw_ref[spike_location[bl]:-1])
    
    if spike_location[bl] != spike_location_expected: 
        print('manual entry of spike location') # things are getting weird with finding the spike location. Record prediction but hard code the location
        meas_raw = trans_raw_w_spike[spike_location_expected:-1] / max(trans_raw_w_spike[spike_location_expected:-1]) 
        if d_ref: meas_raw_ref = trans_raw_ref[spike_location_expected:-1] / max(trans_raw_ref[spike_location_expected:-1])
        # plt.plot(trans_raw_w_spike) # verify that all spikes are lined up where you want them to be

    #%% Calculate DCS frequency axis
        
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
        wvn_raw = wvn_raw[:len(meas_raw)]
            
    elif nyq_side == -1:
        nyq_stop = (nyq_range * (nyq_num - 0.5) + fopt) # Nyquist window 0.5 - 1.0
        wvn_raw = np.arange(nyq_start, nyq_stop, -favg) * hz2cm
        wvn_raw = wvn_raw[:len(meas_raw)]
    
        meas_raw = np.flip(meas_raw)
        wvn_raw = np.flip(wvn_raw)
    
    # plt.plot(wvn_raw, meas_raw)
    
    #%% check if we need to fit the data (or if we already saved that info previously)
    
    if not calc_fits: 
        
        f = open(d_conditions, 'rb')
        if d_ref: [bl_conditions, bl_conditions_ref] = pickle.load(f)
        else: [bl_conditions] = pickle.load(f)
        f.close() 
        
    else: 

        #%% spectrally trim and normalize spectra, convert to TD in preparation for fitting it
        
        [istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_range_fit, max_prime_factor=50, print_value=False)
        
        wvn_fit = wvn_raw[istart:istop] 
        meas_fit = meas_raw[istart:istop] 
        
        meas_fit_flat = meas_fit / np.poly1d(np.polyfit(wvn_fit, meas_fit, 3))(wvn_fit)
        
        meas_TD_fit = np.fft.irfft(-np.log(meas_fit_flat))
        
        if d_ref: 
            meas_fit_ref = meas_raw_ref[istart:istop]
            meas_fit_flat_ref = meas_fit_ref / np.poly1d(np.polyfit(wvn_fit, meas_fit_ref, 3))(wvn_fit)
            meas_TD_fit_ref = np.fft.irfft(-np.log(meas_fit_flat_ref))
        
        #%% prepare to calculate background water concentration
        
        mod1, pars = td.spectra_single_lmfit('T1')
        
        pars['T1mol_id'].value = 1 # water = 1 (hitran molecular code)
        
        pars['T1pressure'].set(value = Patm / 760, vary = vary_pressure) # pressure in atm (converted from Torr)
        pars['T1molefraction'].set(value = yH2Obg, vary = True) # mole fraction


        if not include_meas_refs or bl < bl_vac_number: # don't process measurements with water in the cell
                    
            if two_background_temps: # process 2 background temperatures
    
                Tguess1 = Tatm
                Tguess2 = Tfurnace[bl]
    
                Tboundary1 = (Tatm+Tfurnace[bl])/2 + 50 # split at average, add a buffer for when they are close
                Tboundary2 = Tboundary1 - 50
    
                pathlength1 = pathlength_4f
                pathlength2 = pathlength_furnace

                pars['T1pathlength'].set(value = pathlength1, vary = False) # pathlength in cm
                pars['T1temperature'].set(value = Tguess1, vary = True, max=Tboundary1, min=100) # [K], for 2T this is 4f temperature (must be lower than furnace temperature)


                mod2, pars2 = td.spectra_single_lmfit('T2')
    
                pars2['T2mol_id'].value = 1  # water = 1 (hitran molecular code)
    
                pars2['T2pressure'].expr = 'T1pressure' # testing linked pressure for now
                pars2['T2molefraction'].set(value=yH2Obg, vary=True)  # mole fraction
    
                pars2['T2pathlength'].set(value = pathlength2, vary = False) # pathlength in cm
                pars2['T2temperature'].set(value = Tguess2, vary = True, min=Tboundary2, max=3000) # [K], furnace temperature (must be higher than 4f temperature)
    
                mod = mod1 + mod2
                pars.update(pars2)

            else: # only process one background temperatures
                
                Tguess1 = np.mean([Tatm,Tfurnace[bl]])
                Tboundary1 = Tatm + (Tatm+Tfurnace[bl])/2 + 50
                pathlength1 = pathlength_4f + pathlength_furnace
        
                pars['T1pathlength'].set(value = pathlength1, vary = False) # pathlength in cm
                pars['T1temperature'].set(value = Tguess1, vary = True, max=Tboundary1) # [K], for 2T this is 4f temperature (must be lower than furnace temperature)
           

        # model_TD_fit = mod.eval(xx=wvn_fit, params=pars, name='H2O')
        # model_fit = np.exp(-np.real(np.fft.rfft(model_TD_fit)))
        
        # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        # plt.title(d_file_meas)
        # plt.plot(wvn_fit, meas_fit_flat, label='raw BG signal '+str(bl))
        # plt.plot(wvn_fit, model_fit, label='model for '+str(bl))
        # plt.xlabel('wavenumber')
        # plt.ylabel('normalized laser signal')
        # plt.legend()
        
        # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        # plt.title(d_file_meas)
        # plt.plot(meas_TD_fit, label='BG')
        # plt.plot(model_TD_fit, label='model')
        # plt.plot(model_TD_fit - meas_TD_fit, label='model - BG')
        # plt.ylabel('time domain response')
        # plt.legend()
        
        #%% run the model, then compile and save background conditions to save time later
        
        if not include_meas_refs or bl < bl_vac_number:
        
            print('fitting background conditions')
            if two_background_temps: print('\t\tfor two temperatures (this will probably take a while)')

            if bl in [0,1,2,3]: bl_which = 0
            elif bl == 4: bl_which = 1
            else: bl_which = 2
            weight = td.weight_func(len(wvn_fit), bl_start, etalons=bl_etalons[bl_which])

            bg_fit = mod.fit(meas_TD_fit, xx=wvn_fit, params=pars, weights=weight)
            # meas_noBL = td.plot_fit(wvn_fit, bg_fit)

            yH2Obg1_fit = bg_fit.params['T1molefraction'].value
            yH2Obg1_fit_unc = bg_fit.params['T1molefraction'].stderr
            
            P1_fit = bg_fit.params['T1pressure'].value * 760
            try: P1_fit_unc = bg_fit.params['T1pressure'].stderr * 760
            except: P1_fit_unc = bg_fit.params['T1pressure'].stderr # can't multiple NAN by a number
            
            T1_fit = bg_fit.params['T1temperature'].value
            T1_fit_unc = bg_fit.params['T1temperature'].stderr
            
            shift1_fit = bg_fit.params['T1shift'].value
            shift1_fit_unc = bg_fit.params['T1shift'].stderr
            
            pl1_fit = bg_fit.params['T1pathlength'].value
            
            if two_background_temps: 
                
                yH2Obg2_fit = bg_fit.params['T2molefraction'].value
                yH2Obg2_fit_unc = bg_fit.params['T2molefraction'].stderr
                
                P2_fit = bg_fit.params['T2pressure'].value * 760
                try: P2_fit_unc = bg_fit.params['T2pressure'].stderr * 760
                except: P2_fit_unc = bg_fit.params['T2pressure'].stderr # can't multiple NAN by a number
                
                T2_fit = bg_fit.params['T2temperature'].value
                T2_fit_unc = bg_fit.params['T2temperature'].stderr
                
                shift2_fit = bg_fit.params['T2shift'].value
                shift2_fit_unc = bg_fit.params['T2shift'].stderr
                
                pl2_fit = bg_fit.params['T2pathlength'].value
                
                bl_conditions[bl,:] = [yH2Obg1_fit, yH2Obg1_fit_unc, P1_fit, P1_fit_unc, T1_fit, T1_fit_unc, shift1_fit, shift1_fit_unc, pl1_fit, 
                                       yH2Obg2_fit, yH2Obg2_fit_unc, P2_fit, P2_fit_unc, T2_fit, T2_fit_unc, shift2_fit, shift2_fit_unc, pl2_fit]
            
            else: 
                
                bl_conditions[bl,:] = [yH2Obg1_fit, yH2Obg1_fit_unc, P1_fit, P1_fit_unc, T1_fit, T1_fit_unc, shift1_fit, shift1_fit_unc, pl1_fit]
            
            if save_results: 
                f = open(d_conditions, 'wb')
                pickle.dump([bl_conditions], f)
                f.close() 
                
        #%% do the same thing again if you have a reference channel
        if d_ref: 
            
            print('fitting background conditions for reference channel')
            
            pars['T1pressure'].set(value = Patm / 760, vary = True) # pressure in atm (converted from Torr)
            pars['T1molefraction'].set(value = yH2Obg, vary = True) # mole fraction
            
            pars['T1pathlength'].set(value = pathlength_4f, vary = False) # pathlength in cm
            pars['T1temperature'].set(value = Tatm, vary = True) # temperature in K
            
            # model_TD_fit = mod.eval(xx=wvn_fit, params=pars, name='H2O')
            # model_fit = np.exp(-np.real(np.fft.rfft(model_TD_fit)))
            
            # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            # plt.title(d_file_meas)
            # plt.plot(wvn_fit, meas_fit_flat_ref, label='raw BG signal '+str(bl))
            # plt.plot(wvn_fit, model_fit, label='model for '+str(bl))
            # plt.xlabel('wavenumber')
            # plt.ylabel('normalized laser signal')
            # plt.legend()
            
            # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            # plt.title(d_file_meas)
            # plt.plot(meas_TD_fit_ref, label='BG')
            # plt.plot(model_TD_fit, label='model')
            # plt.plot(model_TD_fit - meas_TD_fit_ref, label='model - BG')
            # plt.ylabel('time domain response')
            # plt.legend()
                    
            bl_which = 2
            weight = td.weight_func(len(wvn_fit), bl_start, etalons=bl_etalons[bl_which])
            
            bg_fit = mod1.fit(meas_TD_fit_ref, xx=wvn_fit, params=pars, weights=weight)
            # meas_noBL = td.plot_fit(wvn_fit, bg_fit)

            yH2Obg_fit = bg_fit.params['T1molefraction'].value
            yH2Obg_fit_unc = bg_fit.params['T1molefraction'].stderr
            
            P_fit = bg_fit.params['T1pressure'].value * 760
            try: P_fit_unc = bg_fit.params['T1pressure'].stderr * 760
            except: P_fit_unc = bg_fit.params['T1pressure'].stderr # can't multiply NAN by a number
            
            T_fit = bg_fit.params['T1temperature'].value
            T_fit_unc = bg_fit.params['T1temperature'].stderr
            
            shift_fit = bg_fit.params['T1shift'].value
            shift_fit_unc = bg_fit.params['T1shift'].stderr
            
            pl_fit = bg_fit.params['T1pathlength'].value
            
            bl_conditions_ref[bl,:] = [yH2Obg_fit, yH2Obg_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc, pl_fit]
            
            if save_results: 
                f = open(d_conditions, 'wb')
                pickle.dump([bl_conditions, bl_conditions_ref], f)
                f.close()   
        
    # %% generate the model for the spectrum to be subtracted off of the entire BL measurement

    wvn2_range_BL_buffer = [wvn2_range_BL[0]-wvn_buffer, wvn2_range_BL[1]+wvn_buffer]
    
    [istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_range_BL_buffer, max_prime_factor=50, print_value=False)
    
    wvn = wvn_raw[istart:istop]
    meas = meas_raw[istart:istop]
    meas_raw_trim = meas_raw[istart:istop]

    meas_TD = np.fft.irfft(-np.log(meas))
    
    if d_ref: 
        meas_ref = meas_raw_ref[istart:istop]
        meas_raw_trim_ref = meas_raw_ref[istart:istop]
        meas_TD_ref = np.fft.irfft(-np.log(meas_ref))
    
    #%% check if we need to generate the data (or if we already saved that info previously)
    if not calc_background: 
        
        f = open(d_bgcorrected, 'rb')
        if d_ref: [meas_bg_all, meas_raw_all, meas_bg_all_ref, meas_raw_all_ref, wvn_all] = pickle.load(f) 
        else: [meas_bg_all, meas_raw_all, wvn_all] = pickle.load(f) 
        f.close() # note that wvn only applies to last dataset processed (for reference)
        
    else:       
    
        mod, pars = td.spectra_single_lmfit()
    
        pars['mol_id'].value = 1 # water = 1 (hitran molecular code)
        
        meas_TD_bg = meas_TD.copy()
        
        if not include_meas_refs or bl < bl_vac_number:
            
            if two_background_temps: num_model = 2
            else:  num_model = 1
            
            if bl in [0, 3, 10, 12, 15, 16, 19, 21, 23, 25, 29]: 
                plt.figure()
                plt.title('baseline #{}'.format(bl))
            
            for i in range(num_model): 
    
                pars['pathlength'].set(value = bl_conditions[bl,8+i*9]) # pathlength in cm
            
                pars['molefraction'].value = bl_conditions[bl,0+i*9] # mole fraction
                pars['pressure'].value = bl_conditions[bl,2+i*9] / 760 # pressure in atm 
                pars['temperature'].value = bl_conditions[bl,4+i*9] # temperature in K
                pars['shift'].value = bl_conditions[bl,6+i*9] # spectral shift in cm-1
                
                print('generating spectrum at background conditions')
                model_TD_bg = mod.eval(xx=wvn, params=pars, name='H2O')
                meas_TD_bg = meas_TD_bg - model_TD_bg # if there are 2 T's, this will remove both
                meas_bg = np.exp(-np.real(np.fft.rfft(meas_TD_bg)))
                
                if bl in [0, 3, 10, 12, 15, 16, 19, 21, 23, 25, 29]: 
                    plt.plot(wvn, np.exp(-np.real(np.fft.rfft(model_TD_bg))), label=i) 
                    plt.legend()
                
        
        #%% do the same thing again if you have a reference channel

        if d_ref: 
            
            pars['pathlength'].set(value = bl_conditions_ref[bl,8]) # pathlength in cm
        
            pars['molefraction'].value = bl_conditions_ref[bl,0] # mole fraction
            pars['pressure'].value = bl_conditions_ref[bl,2] / 760 # pressure in atm 
            pars['temperature'].value = bl_conditions_ref[bl,4] # temperature in K
            pars['shift'].value = bl_conditions_ref[bl,6] # spectral shift in cm-1
            
            print('generating reference spectrum at background conditions')
            model_TD_bg = mod.eval(xx=wvn, params=pars, name='H2O')
            meas_TD_bg = meas_TD_ref - model_TD_bg # remove background absorption from reference channel
            meas_bg_ref = np.exp(-np.real(np.fft.rfft(meas_TD_bg)))
        

        #%% make sure things can be combined into a single 2D list 

        if len(wvn) != 199434: 
            if len(wvn) < 199434: 
                print('array is too short ' + str(len(wvn)))
                if not include_meas_refs or bl < bl_vac_number:
                    meas_bg = np.append(meas_bg, meas_bg[-1]) # not sure what happened here, as a bit too short
                    meas_raw_trim = np.append(meas_raw_trim, meas_raw_trim[-1])
                if d_ref: 
                    meas_bg_ref = np.append(meas_bg_ref, meas_bg_ref[-1]) # not sure what happened here, as a bit too short
                    meas_raw_trim_ref = np.append(meas_raw_trim_ref, meas_raw_trim_ref[-1])
                wvn = np.append(wvn, wvn[-1])
                
            if len(wvn) > 199434: 
                print('array is too long ' + str(len(wvn)))
                if not include_meas_refs or bl < bl_vac_number:
                    meas_bg = meas_bg[0:-1] # not sure what happened here, as a bit too long
                    meas_raw_trim = meas_raw_trim[0:-1]
                if d_ref: 
                    meas_bg_ref = meas_bg_ref[0:-1] # not sure what happened here, as a bit too long
                    meas_raw_trim_ref = meas_raw_trim_ref[0:-1]
                wvn = wvn[0:-1]
        
        if not include_meas_refs or bl < bl_vac_number:
            meas_bg_all[bl,:]  = meas_bg
            meas_raw_all[bl,:] = meas_raw_trim
        if d_ref: 
            meas_bg_all_ref[bl,:]  = meas_bg_ref
            meas_raw_all_ref[bl,:] = meas_raw_trim_ref
        wvn_all[bl,:] = wvn

        if save_results: 
            f = open(d_bgcorrected, 'wb')
            if d_ref: pickle.dump([meas_bg_all, meas_raw_all, meas_bg_all_ref, meas_raw_all_ref, wvn_all], f)
            else: pickle.dump([meas_bg_all, meas_raw_all, wvn_all], f)
            f.close() 
        

#%% remove noise spikes before smoothing    

    meas_spike_all[bl,:] = meas_bg_all[bl,:].copy()
    
    if d_ref: 
        meas_spike_all_ref[bl,:] = meas_bg_all_ref[bl,:].copy()

    if remove_spikes: 
        print('********** warning - this section of code was tested on a case by case basis **********')
        print('********** to remove noise spikes in the vacuum data **********')
        print('********** proceed with caution if you are using it **********')
    
        for j in range(len(meas_spike_all[bl,:])): # second index = which wavenumber
            
            point_ij = meas_bg_all[bl,j]
            points_span = meas_bg_all[bl,j-spike_points_num:j+spike_points_num]
            points_avg = np.mean(points_span)
            points_std = np.std(points_span) 
            
            try: 
                if abs(point_ij-points_avg) > spike_threshold*points_std: 
                    meas_spike_all[bl,j] = meas_spike_all[bl,j-1]
            except: pass
        
            if d_ref: 
                
                point_ij = meas_bg_all_ref[bl,j]
                points_span = meas_bg_all_ref[bl,j-spike_points_num:j+spike_points_num]
                points_avg = np.mean(points_span)
                points_std = np.std(points_span) 
                
                try: 
                    if abs(point_ij-points_avg) > spike_threshold*points_std: 
                        meas_spike_all_ref[bl,j] = meas_spike_all[bl,j-1]
                except: pass


    index_all_final[bl,:] = td.bandwidth_select_td(wvn, wvn2_range_BL, max_prime_factor=50, print_value=False) # remove buffer


# %% filter the data and remove the buffer
        
if not all(index_all_final[:,0] == index_all_final[0,0]) or not all(index_all_final[:,1] == index_all_final[0,1]): 
    please = stophere # the final arrays aren't the same length, something is off (probably a rounding thing, need to add an edge case)

b, a = signal.butter(forderLP, fcutoffLP)
meas_filt_all = signal.filtfilt(b, a, meas_spike_all, axis=1)
if d_ref: meas_filt_all_ref = signal.filtfilt(b, a, meas_spike_all_ref, axis=1)

istart = int(index_all_final[0,0])
istop = int(index_all_final[0,1])

wvn_all_final = wvn_all[:,istart:istop]
meas_raw_all_final = meas_raw_all[:,istart:istop] 
meas_bg_all_final = meas_bg_all[:,istart:istop]
meas_spike_all_final = meas_spike_all[:,istart:istop]
meas_filt_all_final = meas_filt_all[:,istart:istop]

if d_ref: 
    meas_raw_all_final_ref = meas_raw_all_ref[:,istart:istop] 
    meas_bg_all_final_ref = meas_bg_all_ref[:,istart:istop]
    meas_spike_all_final_ref = meas_spike_all_ref[:,istart:istop]
    meas_filt_all_final_ref = meas_filt_all_ref[:,istart:istop]


 #%%

for bl in [0, 3, 10, 12, 15, 16, 19, 21, 23]:
    
    plt.figure()
    plt.title('plot #{}, with threshold of {}'.format(bl,spike_threshold))
    plt.plot(wvn_all_final[bl,:], meas_raw_all_final[bl,:], label='raw')
    plt.plot(wvn_all_final[bl,:], meas_bg_all_final[bl,:], label='bg water removal')
    plt.plot(wvn_all_final[bl,:], meas_spike_all_final[bl,:], label='noise spike removal')
    plt.plot(wvn_all_final[bl,:], meas_filt_all_final[bl,:], label='lowpass filter')
    
    point_diff = np.diff(meas_bg_all_final[bl,:])
    point_std = np.std(np.diff(meas_bg_all_final[bl,:]))

    plt.plot(wvn_all_final[bl,:-1], point_diff/point_std/100, label='differential of raw')
    # plt.plot(wvn_all_final[bl,:], point_diff/point_std/100 + meas_filt_all_final[bl,:-1]+0.2, label='raw')

    plt.plot(wvn_all_final[bl,:], (meas_bg_all_final[bl,:] - meas_spike_all_final[bl,:])*10-0.25)
    
    plt.legend(loc='upper right')


# %% add a row of all 1's (does nothing) for datasets where you want to do nothing (combine after applying separate baselines)

# meas_filt_all_final = np.vstack((meas_filt_all_final,np.ones_like(wvn_all_final[iplot,:]))) # everything with a row of 1's (null BL)


# %% save the filtered data 

d_final = os.path.join(d_folder, 'BL filtered '+ d_filter +'.pckl')

if save_results: 
    f = open(d_final, 'wb')
    pickle.dump([meas_filt_all_final, meas_filt_all_final_ref, wvn_all_final], f)
    f.close() 


