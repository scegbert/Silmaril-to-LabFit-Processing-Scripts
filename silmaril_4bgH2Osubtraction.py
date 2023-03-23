"""

silmaril 4 - bg H2O subtraction

Created on Thu Aug 13 08:42:46 2020

@author: scott

incorporate everything (measurement files, vacuum laser, background water conditions, etc.) and output transmission files
also checks to see how well things match the model

"""

import time 
# time.sleep(60*60*4) # hang out for 4 hours

import numpy as np
import pickle 
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import pldspectrapy as pld
import td_support as td
import linelist_conversions as db

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

# %% dataset specific information

d_type = 'air' # 'pure' or 'air'
d_ref = True
spike_location_expected = 13979
spectrum_length_expected = 190651


which_BL_pure = [0,0,0,0,0,0,0,0, 10,10,10,10,10, 29,12,12,12,29, 15,15,16,16,16, 23,23,23,21,21, 25] 
#                0,1,2,3,4,5,6,7   8, 9,10,11,12  13,14,15,16,17  18,19,20,21,22  23,24,25,26,27, 28
which_BL_air = [3,3,3,3,3,3,3,3, 10,10,10,10,10, 12,12,12,12,12, 16,16,16,16,16, 19,19,19,19,19, 25]
#               0,1,2,3,4,5,6,7   8, 9,10,11,12  13,14,15,16,17  18,19,20,21,22  23,24,25,26,27, 28
bl_number = 30 # there are 30 of them (0-29)


BL_fcutoff = '0.030'
spike_location_expected = 13979


check_bl = False # look at a bunch of BL's to try to find the right one
two_BG_temps = True # number of background temperatures to subtract (False==1, True==2)
check_fit = True # fit measurments against HITRAN database (2020, 2016, and Paul)
save_file = check_fit

nyq_side = 1 # which side of the Nyquist window are you on? + (0 to 0.25) or - (0.75 to 0)

wvn2_processing = [6500, 7800]
wvn2_spectroscopy = [] # range we will send to labfit
wvn2_concentration = [[7010.2,7010.7],[7012.45,7012.95],[7026.35,7027],[7038.1,7038.8],[7041.8,7042.8],
                      [7046.9,7048.3],[7070.1,7071.2],[7071.05,7072.3],[7085.5,7086.3],[7093.7,7095.3],[7104.3,7105.3],
                      [7116.6,7118.6],[7138.4,7140.4],[7160.95,7162.1],[7167.6,7169.6],[7193.6,7196],[7198.5,7200],
                      [7201.5,7203.5],[7204.4,7206.2],[7212.3,7213.4],[7214.9,7215.9],[7215.9,7216.9],[7218.6,7220.3],
                      [7241.55,7245],[7293.05,7295.6],[7311.4,7313.5],[7326.9,7329.3],[7338.5,7341.4],[7347.6,7349.2],
                      [7355, 7356.6],[7374.1,7376.17],[7389,7390.6],[7396.7,7398.5],[7405.5,7406.3],[7412.8,7413.4],
                      [7413.3,7414.4],[7415.8,7416.6],[7417.2,7418.3]] # all low-uncertainty features

wvn2_etalon = [6630,7570] # some kind of etalon range with vacuum scans from December 2020 water data. can't remember why I included this

d_base_pure = ['300 K _5 T', '300 K 1 T',  '300 K 1_5 T','300 K 2 T',  '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
               '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
               '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
               '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
               '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T', 
               '1300 K 16 T']

d_base_air = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
              '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
              '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
              '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
              '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T', 
              '1300 K 600 T']

pathlength = 91.4 # pathlength inside the quartz cell (double passed)

which_BG_pure = [0,0,0,0,0,0,0,0, 10,10,10,10,10, 29,12,12,12,29, 15,15,16,16,16, 23,23,23,21,21, 23] # don't mess with this one (determined by date of data collection)
which_BG_air = [3,3,3,3,3,3,3,3, 10,10,10,10,10, 12,12,12,12,12, 16,16,16,16,16, 19,19,19,19,19, 23] # don't mess with this one (determined by date of data collection)

bl_cutoff = 101

if d_type == 'pure': 

    y_h2o = 1
    calc_yh2o = False
    
    d_base = d_base_pure
    which_BL = which_BL_pure
    which_BG = which_BG_pure
    
    vary_P = True
    vary_yh2o = False
    
    d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'
    which_conditions = 'Pure Water P & T.pckl'
    
    which_ref_start = bl_number # file to start with when identifying reference channel files
    
    wvn2_fit = [6699.6, 7041.1] # avoiding big features (saturation issues), using bin breaks

    
elif d_type == 'air':
    
    y_h2o_all = [0.0189748, 0.0191960, 0.0192649, 0.0193174, 0.0193936, 0.0194903, 0.0195316, 0.0194732, 
                 0.0193572, 0.0193187, 0.0192066, 0.0192580, 0.0195697, 
                 0.0189490, 0.0190159, 0.0189894, 0.0189217, 0.0189221, 
                 0.0186053 ,0.0189104 ,0.0187065, 0.0185842, 0.0185690, 
                 0.0191551, 0.0195356, 0.0192415, 0.0187509, 0.0188582, 
                 0.0193093] # calculated using 38 features (listed above)
   
    calc_yh2o = False # probably don't want to fit against model yet (set to False)

    d_base = d_base_air
    which_BL = which_BL_air
    which_BG = which_BG_air

    vary_P = False
    vary_yh2o = True
    
    d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water'
    which_conditions = 'Air Water P & T.pckl'
    
    which_ref_start = bl_number + len(d_base_pure) # file to start with when identifying reference channel files

    wvn2_fit = [6699.6, 7177.4] # avoiding big features (saturation issues), using bin breaks

d_vac = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'

spike_location = np.zeros(len(d_base), dtype=int)
output2020 = np.zeros((len(d_base),8))
output2016 = np.zeros((len(d_base),8))
outputPaul = np.zeros((len(d_base),8))

if calc_yh2o: 
    bl_cutoff_h2o = 2 # TD fit BL cutoff for calculating yh2o for a single feature (very narrow)
    output_yh20 = np.zeros((len(d_base),len(wvn2_concentration)))
    output_P = np.zeros((len(d_base),len(wvn2_concentration)))
    output_yh20P = np.zeros((len(d_base),len(wvn2_concentration)))
    output_Pyh2o = np.zeros((len(d_base),len(wvn2_concentration)))
    output_shift = np.zeros((len(d_base),len(wvn2_concentration)))
    output_yh20_test = {}
    
    path_yh2o_load = r'\\data - HITRAN 2020\\'
    path_yh2o_save = r'\\data - temp\\'
    df_yh2o = db.par_to_df(os.path.abspath('') + path_yh2o_load + 'H2O.par')

#%% load in measured conditions (P and T) 

d_load = os.path.join(d_meas, which_conditions)

f = open(d_load, 'rb')
[P_all, T_all] = pickle.load(f)
f.close() 

for which_file in range(len(d_base)): # check with d_base[which_file]
    
    print(d_base[which_file])

    if 'y_h2o_all' in locals(): y_h2o = y_h2o_all[which_file] # if we have values for yh2o (ie air-water)
    
    P = P_all[which_file]
    T = T_all[which_file]
    
    
    # %% Locate phase-corrected measured spectrum
    
    d_load = os.path.join(d_meas, d_base[which_file] + ' pre.pckl')
    
    f = open(d_load, 'rb')
    if d_ref: [trans_raw_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt, trans_raw_ref, trans_snip_ref] = pickle.load(f)
    else: [trans_raw_w_spike, trans_snip, coadds, dfrep, frep1, frep2, ppig, fopt] = pickle.load(f)
    f.close() 
        
    
    # %% Locate and remove spike
        
    spike_location[which_file] = np.argmax(np.abs(np.diff(trans_raw_w_spike, n=5))) + 2 # the nyquist window starts with a large delta function (location set by shift in Silmatil GUI)
    
    meas_raw = trans_raw_w_spike[spike_location[which_file]:-1] / max(trans_raw_w_spike[spike_location[which_file]:-1]) # normalize max to 1
    if d_ref: meas_raw_ref = trans_raw_ref[spike_location[which_file]:-1] / max(trans_raw_ref[spike_location[which_file]:-1])
    
    if spike_location[which_file] != spike_location_expected: 
        # print('manual entry of spike location') # things are getting weird with finding the spike location. Record prediction but hard code the location
        meas_raw = trans_raw_w_spike[spike_location_expected:-1] / max(trans_raw_w_spike[spike_location_expected:-1]) 
        if d_ref: meas_raw_ref = trans_raw_ref[spike_location_expected:-1] / max(trans_raw_ref[spike_location_expected:-1])
        # plt.plot(trans_raw_w_spike) # verify that all spikes are lined up where you want them to be
    
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
        wvn_raw = wvn_raw[:len(meas_raw)]
            
    elif nyq_side == -1:
        nyq_stop = (nyq_range * (nyq_num - 0.5) + fopt) # Nyquist window 0.5 - 1.0
        wvn_raw = np.arange(nyq_start, nyq_stop, -favg) * hz2cm
        wvn_raw = wvn_raw[:len(meas_raw)]
    
        meas_raw = np.flip(meas_raw)
        wvn_raw = np.flip(wvn_raw)
        
    # %% spectrally trim things
    
    [istart, istop] = td.bandwidth_select_td(wvn_raw, wvn2_processing, max_prime_factor=50, print_value=False)
    wvn = wvn_raw[istart:istop]
    
    if len(wvn) != spectrum_length_expected: 
        print('weird length of wvn, changing istop - ' + str(which_file))
        please=stop_here # something is wrong with that file and it will not run
        
    meas = meas_raw[istart:istop] 
    
    # %% Generate HITRAN model at measured conditions (if needed, this takes a minute so let's try to avoid it if possible)
    
    try: 
        
        d_load = os.path.join(d_meas, d_base[which_file] + ' model.pckl')
        
        f = open(d_load, 'rb')
        [model_TD] = pickle.load(f)
        f.close()
    
    except: 
    
        pld.db_begin('data - HITRAN 2020')  # load the linelists into Python
    
        meas_mod, meas_pars = td.spectra_single_lmfit() # this makes a single-path H2O cell path to model using HAPI, which is available online
        meas_pars['mol_id' ].value = 1 # water = 1 (hitran molecular code)
        
        meas_pars['pathlength'].value = pathlength # pathlength in cm
        meas_pars['molefraction'].value = y_h2o # mole fraction
        meas_pars['temperature'].value = T # temperature in K
        meas_pars['pressure'].value = P / 760 # pressure in atm (converted from Torr)
        
        model_TD = meas_mod.eval(xx=wvn, params=meas_pars, name='H2O') # 'H2O' needs to be in db_begin directory
    
        
        d_load = os.path.join(d_meas, d_base[which_file] + ' model.pckl')
        
        f = open(d_load, 'wb')
        pickle.dump([model_TD], f)
        f.close() 
    
    model_abs = np.real(np.fft.rfft(model_TD)) 
    model_trans = np.exp(-model_abs)
        
    # %% load filtered laser baseline spectra
    
    if two_BG_temps: d_load = os.path.join(d_vac, 'BL filtered 1 ' + BL_fcutoff + ' with 2Ts.pckl')
    else:  d_load = os.path.join(d_vac, 'BL filtered 1 ' + BL_fcutoff + '.pckl')
    
    f = open(d_load, 'rb')
    if d_ref: [bl_filt_all, bl_filt_all_ref, wvn_bl] = pickle.load(f)
    else: [bl_filt_all, wvn_bl] = pickle.load(f)
    f.close()
    
    if not check_bl: # we're only looking at one baseline 
        
        bl_filt = bl_filt_all[which_BL[which_file],:]
        
        meas_trans_bl = meas / bl_filt
        
        # plt.figure()
        # plt.plot(wvn, meas_trans_bl)
        # plt.title(d_base[which_file])
            
    #%% check against lots of baseline options if needed
    else: 
        
        if type(which_BL[which_file]) == int: # if there's only 1 option
            
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            plt.plot(wvn, meas / bl_filt_all[which_BL[which_file],:], label='indicated BL')
            plt.plot(wvn, meas / bl_filt_all_ref[which_ref_start+which_file,:], label='reference as BL')
            plt.legend()
        
        else: 
                        
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k'); f1 = plt.gcf().number
            plt.plot(wvn, meas, label='raw water signal')
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k'); f2 = plt.gcf().number
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k'); f4 = plt.gcf().number
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k'); f7 = plt.gcf().number
                
            for i in which_BL[which_file]:
                
                bl_filt = bl_filt_all[i,:]
                
                bl_i = str(i)
                
                plt.figure(f1)
                plt.plot(wvn, bl_filt, linewidth = 1, label='BL ' + bl_i)
        
                plt.xlabel('wavenumber')
                plt.ylabel('normalized laser signal')
                plt.legend(loc = 'upper right')
                
                meas_trans_bl = meas / bl_filt
                
                # intermediate step to check which BL to use
                plt.figure(f2)
                plt.plot(wvn, meas_trans_bl - i/100, linewidth = 1, label='BL ' + bl_i)
                plt.legend(loc = 'upper right')
                                
                meas_abs_bl = -np.log(meas_trans_bl)
                meas_TD_bl = np.fft.irfft(meas_trans_bl)
                
                residual_TD = meas_TD_bl - model_TD
                residual_TD_std = np.std(residual_TD[1000:2000])
                residual_trans = np.exp(-np.real(np.fft.rfft(residual_TD)))
                        
                plt.figure(f4)
                plt.plot(residual_TD + i/700, linewidth = 1, label='BL ' + bl_i + ', ' + "{0:.1E}".format(residual_TD_std))
                plt.ylabel('residual (meas_TD - model_TD) ')
                plt.legend(loc = 'upper right')
                
                [istart, istop] = td.bandwidth_select_td(wvn, wvn2_etalon, max_prime_factor=50, print_value=False)
                
                wvn_etalon = wvn[istart:istop]
                meas_trans_BL_etalon = meas_trans_bl[istart:istop] 
                
                model_trans_etalon = model_trans[istart:istop]
                model_abs_etalon = -np.log(model_trans_etalon)
                model_TD_etalon = np.fft.irfft(model_abs_etalon)
                
                meas_abs_BL_etalon = -np.log(meas_trans_BL_etalon)
                meas_TD_BL_etalon = np.fft.irfft(meas_abs_BL_etalon)
        
                residual_TD_etalon = meas_TD_BL_etalon - model_TD_etalon
                residual_TD_etalon_std = np.std(residual_TD_etalon[1000:2000])
                
                plt.figure(f7)
                plt.plot(residual_TD_etalon + i/70, linewidth = 1, label='BL ' + bl_i + ', ' + "{0:.1E}".format(residual_TD_etalon_std))
                plt.ylabel('residual (meas_TD - model_TD) for etalon region')
                plt.legend(loc = 'upper right')
                    
        please = stophere # no need to continue if you haven't picked a baseline yet
    #%% load in background conditions

    if two_BG_temps: 
        d_load = os.path.join(d_vac, 'BL conditions with 2Ts.pckl')
        num_backgroundTs = 2
    else: 
        d_load = os.path.join(d_vac, 'BL conditions.pckl')
        num_backgroundTs = 1
    
    f = open(d_load, 'rb')
    if d_ref: [bl_conditions, bl_conditions_ref]  = pickle.load(f) 
    else: [bl_conditions]  = pickle.load(f)    
    f.close() 
       
    # %% calculate and remove background water 
   
    try: 
        
        d_load = os.path.join(d_meas, d_base[which_file] + ' model background.pckl') # see if we've already done it
        
        f = open(d_load, 'rb')
        [bg_TD] = pickle.load(f)
        f.close()
    
    except: 
        
        pld.db_begin('data - HITRAN 2020')  # load the linelists into Python
        
        for i in range(num_backgroundTs): 
            
            h2o_BG = bl_conditions[which_BG[which_file],0+i*9]
            P_BG = bl_conditions[which_BG[which_file],2+i*9] 
            T_BG = bl_conditions[which_BG[which_file],4+i*9]
            shift_BG = bl_conditions[which_BG[which_file],6+i*9]
            pathlength_BG = bl_conditions[which_BG[which_file],8+i*9]
            
            bg_mod, bg_pars = td.spectra_single_lmfit() # this makes a single-path H2O cell path to model using HAPI, which is available online
            bg_pars['mol_id' ].value = 1 # water = 1 (hitran molecular code)
            
            bg_pars['pathlength'].value = pathlength_BG # pathlength in cm
            bg_pars['molefraction'].value = h2o_BG # mole fraction
            bg_pars['temperature'].value = T_BG # temperature in K
            bg_pars['pressure'].value = P_BG / 760 # pressure in atm (converted from Torr)
            bg_pars['shift'].value = shift_BG # pressure in atm (converted from Torr)
            
            if i == 0: 
                
                bg_TD1 = bg_mod.eval(xx=wvn, params=bg_pars, name='H2O') # 'H2O' needs to be in db_begin directory
                bg_TD = bg_TD1.copy()
                
            elif i == 1: 
                
                bg_TD2 = bg_mod.eval(xx=wvn, params=bg_pars, name='H2O') # 'H2O' needs to be in db_begin directory
                bg_TD += bg_TD2
        
            d_load = os.path.join(d_meas, d_base[which_file] + ' model background.pckl')
        
        f = open(d_load, 'wb')
        pickle.dump([bg_TD], f)
        f.close() 
    
    bg_abs = np.real(np.fft.rfft(bg_TD))
    bg_trans = np.exp(-bg_abs)
    
    meas_abs_bl = -np.log(meas_trans_bl)
    meas_TD_bl = np.fft.irfft(meas_abs_bl)
    
    meas_abs_bg = meas_abs_bl - bg_abs
    meas_trans_bg = np.exp(-meas_abs_bg)
    meas_TD_bg = np.fft.irfft(meas_abs_bg)
    
    change_bg = meas_trans_bl - meas_trans_bg
    
    # %% calculate water concentration in the cell (if desired)
   
    if calc_yh2o: 
        
        output_yh20_test[d_base[which_file]] = np.zeros((len(wvn2_concentration),6))   
        
        for wvn2 in wvn2_concentration: 
            
            print('     {}       {}'.format(d_base[which_file], wvn2))
            
            df_yh2o_iter = df_yh2o[(df_yh2o.nu > wvn2[0] - 2) & (df_yh2o.nu < wvn2[1] + 2) & (df_yh2o.local_iso_id == 1)]
                        
            db.df_to_par(df_yh2o_iter.reset_index(), par_name='H2O_yh2o', save_dir=os.path.abspath('')+path_yh2o_save, print_name=False)
    
            pld.db_begin('data - temp')  # load the linelists into Python
        
            [istart, istop] = td.bandwidth_select_td(wvn, wvn2, max_prime_factor=50, print_value=False)
            
            wvn_fit = wvn[istart:istop]
            meas_trans_bg_fit = meas_trans_bg[istart:istop] # already normalized near 1   
            
            # plt.plot(wvn_fit, meas_trans_bg_fit)
            
            meas_TD_bg_fit = np.fft.irfft(-np.log(meas_trans_bg_fit))
              
            fit_mod2020, fit_pars2020 = td.spectra_single_lmfit()
            
            fit_pars2020['mol_id'].value = 1 # water = 1 (hitran molecular code)
            fit_pars2020['pathlength'].set(value = pathlength, vary = False) # pathlength in cm
            fit_pars2020['pressure'].set(value = P / 760, vary = True, max=3) # pressure in atm (converted from Torr)
            fit_pars2020['temperature'].set(value = T, vary = False) # temperature in K
    
            fit_pars2020['molefraction'].set(value = y_h2o*0.999, vary = True) # mole fraction
            
            r'''
            model_TD_fit2020 = fit_mod2020.eval(xx=wvn_fit, params=fit_pars2020, name='H2O_yh2o') # used to check baseline decision
            
            plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            plt.plot(model_TD_fit2020, label='model')
            plt.plot(meas_TD_bg_fit, label='BG')
            plt.ylabel('time domain response')
            plt.legend()
            r'''
            
            weight = td.weight_func(len(wvn_fit), bl_cutoff_h2o, etalons = [])
            
            meas_fit_bg2020 = fit_mod2020.fit(meas_TD_bg_fit, xx=wvn_fit, params=fit_pars2020, weights=weight, name='H2O_yh2o')
            # td.plot_fit(wvn_fit, meas_fit_bg2020, plot_td = False)
            
            yh2o_fit = meas_fit_bg2020.params['molefraction'].value
            yh2o_fit_unc = meas_fit_bg2020.params['molefraction'].stderr
            
            P_fit = meas_fit_bg2020.params['pressure'].value * 760
            try: P_fit_unc = meas_fit_bg2020.params['pressure'].stderr * 760
            except: P_fit_unc = meas_fit_bg2020.params['pressure'].stderr # can't multiple NAN by a number
            
            shift_fit = meas_fit_bg2020.params['shift'].value
            shift_fit_unc = meas_fit_bg2020.params['shift'].stderr
            
            output_yh20[which_file, wvn2_concentration.index(wvn2)] = yh2o_fit
            output_P[which_file, wvn2_concentration.index(wvn2)] = P_fit
            
            output_shift[which_file, wvn2_concentration.index(wvn2)] = shift_fit / hz2cm / 1e6 # in MHz
            
            output_yh20P[which_file, wvn2_concentration.index(wvn2)] = yh2o_fit * P_fit / P # adjusted yh2o at known pressure
            output_Pyh2o[which_file, wvn2_concentration.index(wvn2)] = yh2o_fit * P_fit / y_h2o # adjust pressure at known concentration
            
            output_yh20_test[d_base[which_file]][wvn2_concentration.index(wvn2),:] = [yh2o_fit, yh2o_fit_unc, P_fit, P_fit_unc, shift_fit, shift_fit_unc]

    # %% check values by fitting for them against HITRAN 2020 database
    
    if check_fit:
        
        pld.db_begin('data - HITRAN 2020')  # load the linelists into Python
        
        [istart, istop] = td.bandwidth_select_td(wvn, wvn2_fit, max_prime_factor=50, print_value=False)
        
        wvn_fit = wvn[istart:istop]
        meas_trans_bg_fit = meas_trans_bg[istart:istop] # already normalized near 1   
    
        meas_TD_bg_fit = np.fft.irfft(-np.log(meas_trans_bg_fit))
          
        fit_mod2020, fit_pars2020 = td.spectra_single_lmfit() # sd = True is an option
        
        fit_pars2020['mol_id'].value = 1 # water = 1 (hitran molecular code)
        fit_pars2020['pathlength'].set(value = pathlength, vary = False) # pathlength in cm

        fit_pars2020['molefraction'].set(value = y_h2o, vary = vary_yh2o) # mole fraction
        fit_pars2020['pressure'].set(value = P / 760, vary = vary_P) # pressure in atm (converted from Torr)
        
        fit_pars2020['temperature'].set(value = T, vary = True) # temperature in K
        
        model_TD_fit2020 = fit_mod2020.eval(xx=wvn_fit, params=fit_pars2020, name='H2O') # used to check baseline decision
        
        r'''
        plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        plt.plot(model_TD_fit2020, label='model')
        plt.plot(meas_TD_bg_fit, label='BG')
        plt.ylabel('time domain response')
        plt.legend()
        r'''
        
        weight = td.weight_func(len(wvn_fit), bl_cutoff, etalons = [])
        
        meas_fit_bg2020 = fit_mod2020.fit(meas_TD_bg_fit, xx=wvn_fit, params=fit_pars2020, weights=weight)
        # td.plot_fit(wvn_fit, meas_fit_bg2020)
        
        yh2o_fit = meas_fit_bg2020.params['molefraction'].value
        yh2o_fit_unc = meas_fit_bg2020.params['molefraction'].stderr
        
        P_fit = meas_fit_bg2020.params['pressure'].value * 760
        try: P_fit_unc = meas_fit_bg2020.params['pressure'].stderr * 760
        except: P_fit_unc = meas_fit_bg2020.params['pressure'].stderr # can't multiple NAN by a number
    
        T_fit = meas_fit_bg2020.params['temperature'].value
        T_fit_unc = meas_fit_bg2020.params['temperature'].stderr
        
        shift_fit = meas_fit_bg2020.params['shift'].value
        shift_fit_unc = meas_fit_bg2020.params['shift'].stderr
        
        output2020[which_file,:] = [yh2o_fit, yh2o_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc]
        
        model_abs_fit2020 = np.real(np.fft.rfft(fit_mod2020.eval(xx=wvn, params=fit_pars2020, name='H2O')))
        
        model_trans_fit2020 = np.exp(-model_abs_fit2020)
        model_TD_fit2020 = np.fft.irfft(model_abs_fit2020)
        
    # %% check values by fitting for them against HITRAN 2016 database
        

        pld.db_begin('data - HITRAN 2016')  # load the 2016 linelist into Python
        fit_mod2016, fit_pars2016 = td.spectra_single_lmfit()
        
        fit_pars2016['mol_id'].value = 1 # water = 1 (hitran molecular code)
        fit_pars2016['pathlength'].set(value = pathlength, vary = False) # pathlength in cm
        
        fit_pars2016['molefraction'].set(value = y_h2o, vary = vary_yh2o) # mole fraction
        fit_pars2016['pressure'].set(value = P / 760, vary = vary_P) # pressure in atm (converted from Torr)
        
        fit_pars2016['temperature'].set(value = T, vary = True) # temperature in K
        
        meas_fit_bg2016 = fit_mod2016.fit(meas_TD_bg_fit, xx=wvn_fit, params=fit_pars2016, weights=weight)
        #td.plot_fit(wvn_fit, meas_fit_bg2016)
        
        yh2o_fit = meas_fit_bg2016.params['molefraction'].value
        yh2o_fit_unc = meas_fit_bg2016.params['molefraction'].stderr
        
        P_fit = meas_fit_bg2016.params['pressure'].value * 760
        try: P_fit_unc = meas_fit_bg2016.params['pressure'].stderr * 760
        except: P_fit_unc = meas_fit_bg2016.params['pressure'].stderr # can't multiple NAN by a number
    
        T_fit = meas_fit_bg2016.params['temperature'].value
        T_fit_unc = meas_fit_bg2016.params['temperature'].stderr
        
        shift_fit = meas_fit_bg2016.params['shift'].value
        shift_fit_unc = meas_fit_bg2016.params['shift'].stderr
        
        output2016[which_file,:] = [yh2o_fit, yh2o_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc]
    
        model_abs_fit2016 = np.real(np.fft.rfft(fit_mod2016.eval(xx=wvn, params=fit_pars2016, name='H2O')))
        
        model_trans_fit2016 = np.exp(-model_abs_fit2016)
        model_TD_fit2016 = np.fft.irfft(model_abs_fit2016)
    
    
    # %% finally, fit against paul's database
    
        pld.db_begin('data - Paul')  # load Paul's linelist into Python
        fit_modPaul, fit_parsPaul = td.spectra_single_lmfit() 
        
        fit_parsPaul['mol_id'].value = 1 # water = 1 (hitran molecular code)
        fit_parsPaul['pathlength'].set(value = pathlength, vary = False) # pathlength in cm

        fit_parsPaul['molefraction'].set(value = y_h2o, vary = vary_yh2o) # mole fraction
        fit_parsPaul['pressure'].set(value = P / 760, vary = vary_P) # pressure in atm (converted from Torr)

        fit_parsPaul['temperature'].set(value = T, vary = True) # temperature in K
        
        meas_fit_bg_Paul = fit_modPaul.fit(meas_TD_bg_fit, xx=wvn_fit, params=fit_parsPaul, weights=weight)
        #td.plot_fit(wvn_fit, meas_fit_bg_Paul)
        
        yh2o_fit = meas_fit_bg_Paul.params['molefraction'].value
        yh2o_fit_unc = meas_fit_bg_Paul.params['molefraction'].stderr 
                
        P_fit = meas_fit_bg_Paul.params['pressure'].value * 760
        try: P_fit_unc = meas_fit_bg_Paul.params['pressure'].stderr * 760
        except: P_fit_unc = meas_fit_bg_Paul.params['pressure'].stderr # can't multiple NAN by a number
    
        T_fit = meas_fit_bg_Paul.params['temperature'].value
        T_fit_unc = meas_fit_bg_Paul.params['temperature'].stderr
        
        shift_fit = meas_fit_bg_Paul.params['shift'].value
        shift_fit_unc = meas_fit_bg_Paul.params['shift'].stderr
        
        outputPaul[which_file,:] = [yh2o_fit, yh2o_fit_unc, P_fit, P_fit_unc, T_fit, T_fit_unc, shift_fit, shift_fit_unc]
    
        model_abs_fitPaul = np.real(np.fft.rfft(fit_modPaul.eval(xx=wvn, params=fit_parsPaul, name='H2O')))
        
        model_trans_fitPaul = np.exp(-model_abs_fitPaul)
        model_TD_fitPaul = np.fft.irfft(model_abs_fitPaul)
    
        # %% compare transmission spectra

        # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        # plt.plot(wvn, model_trans*100, label='model at measured furnace conditions')
        # if check_fit:
        #     plt.plot(wvn, model_trans_fit2020*100, label='model at fit furnace conditions')
        # # plt.plot(wvn, meas_trans_BL, label='measured - BL correction only')
        # plt.plot(wvn, meas_trans_bg*100, label='measured w/ BL and BG corrections')
        # plt.plot(wvn, (meas_trans_bg-model_trans_fit2020)*100 + 105, label='measured w/ BL and BG corrections')
        # plt.plot(wvn, bg_trans*100, label='model of background water absorption')
        
        # if wvn2_fit is not None:
        #     plt.axvline(x=wvn2_fit[0],color='k', linestyle='-.', label='fit region')
        #     plt.axvline(x=wvn2_fit[1],color='k', linestyle='-.')
        
        # plt.xlabel('wavenumber')
        # plt.ylabel('% transmission')
        # plt.legend(loc='lower right')

        
        # %% compare residuals for transmission spectra
        
        # residual_bg2model = meas_trans_bg - model_trans
        # if check_fit:
        #     residual_bg2modelfit = meas_trans_bg - model_trans_fit2020
        
        # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        # if check_fit:
        #     plt.plot(wvn, residual_bg2modelfit, label='BG - model at fit conditions')
        # plt.plot(wvn, residual_bg2model, label='BG - model at meas conditions')
        # plt.plot(wvn, change_bg, label=' BL and LP filter - BL, Filter, BG corrections')
        
        # if wvn2_fit is not None:
        #     plt.axvline(x=wvn2_fit[0],color='k', linestyle='-.', label='fit region')
        #     plt.axvline(x=wvn2_fit[1],color='k', linestyle='-.')
        
        # plt.xlabel('wavenumber')
        # plt.ylabel('residual')
        # plt.legend()
        
        # %% compare residuals for time domain spectra
        
        # residual_TD_bg2model = meas_TD_bg - model_TD
        # if check_fit:
        #     residual_TD_bg2modelfit = meas_TD_bg - model_TD_fit2020
        
        # plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
        # plt.plot(residual_TD_bg2model, label='BG_TD - model_TD at meas conditions')
        # if check_fit:
        #     plt.plot(residual_TD_bg2modelfit, linewidth=1, label='BG_TD - model_TD at fit conditions')
        
        # plt.ylabel('residual')
        # plt.legend()
    
    
    else: # if not checking fits, create these variables so you can save other stuff
    
        model_trans_fit2020 = 0
        model_trans_fit2016 = 0
        model_trans_fitPaul = 0
    
    # %% save transmission spectrum
    
    fitresults_all = np.hstack((output2020[which_file,:], output2016[which_file,:], outputPaul[which_file,:])) # for just this condition, for saving
   
    d_load = os.path.join(d_meas, d_base[which_file] + ' bg subtraction.pckl')
    
    if save_file: 
        
        f = open(d_load, 'wb')
        # meas_trans_bg = (meas / bl) - H2O_bg, meas_trans_bl = (meas / bl) 
        pickle.dump([meas_trans_bg, meas_trans_bl, wvn, T, P, y_h2o, pathlength, favg, fitresults_all, model_trans_fit2020, model_trans_fit2016, model_trans_fitPaul], f)
        f.close() 

