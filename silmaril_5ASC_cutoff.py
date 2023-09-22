r'''

silmaril 5 - ASC cutoff

prepares transmission data for labfit .ASC file format (includes provision for saturated features that are below a specified cutoff threshold)

r'''



import numpy as np
import pickle 
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

from scipy.constants import speed_of_light
import pldspectrapy as pld
import td_support as td

import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()

import time

# %% dataset specific information

save_data = False
check_bins = True # only look at some of the measurements (high pressure, to check the bin-breaks for features)
d_type = 'air' # 'pure' or 'air'

remove_bg = True # if True, use the transmission file with the background removed, otherwise send the background information to labfit
two_BG_temps = True # only matters if remove_bg = False (trying to put conditions in ASC), not yet prepared for two backgrounds in the ASC file

d_ref = True # there is a reference channel

nyq_side = 1 # which side of the Nyquist window are you on? + (0 to 0.25) or - (0.75 to 0)

if d_type == 'pure': 
    
    d_base = ['300 K _5 T', '300 K 1 T',  '300 K 1_5 T','300 K 2 T',  '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
              '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
              '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
              '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
              '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T', '1300 K 16 T']
    
    if check_bins: 
        d_base = ['300 K 16 T', '500 K 16 T', '700 K 16 T', '900 K 16 T', '1100 K 16 T', '1300 K 16 T']
        # d_base = ['300 K _5 T', '300 K 1 T',  '300 K 1_5 T','300 K 2 T',  '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T']
    
    d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'
    d_meas = r'E:\water database\data - 2021-08\pure water'

    file_number = 1000 # counter to start the save files at (skipped 2000 due to first round)
 
    which_BG = [0,0,0,0,0,0,0,0, 10,10,10,10,10, 29,12,12,12,29, 15,15,16,16,16, 23,23,23,21,21, 23] # don't mess with this one (determined by date of data collection)

    d_save = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water\labfit files'
    d_save = r'E:\water database\data - 2021-08\pure water\labfit files'
    
elif d_type == 'air':

    d_base = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
              '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
              '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
              '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
              '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T', '1300 K 600 T']
    
    if check_bins: 
        d_base = ['300 K 600 T', '500 K 600 T', '700 K 600 T', '900 K 600 T', '1100 K 600 T', '1300 K 600 T']
        # d_base = ['300 K 20 T', '300 K 600 T', '700 K 40 T', '700 K 600 T', '1100 K 40 T', '1100 K 600 T', '1300 K 600 T']
        # d_base = ['1300 K 600 T']

    
    d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water'
    d_meas = r'E:\water database\data - 2021-08\air water'

    file_number = 5000 # counter to start the save files at (skipped 2000 due to first round)

    which_BG = [3,3,3,3,3,3,3,3, 10,10,10,10,10, 12,12,12,12,12, 16,16,16,16,16, 19,19,19,19,19, 23] # don't mess with this one (determined by date of data collection)

    d_save = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water\labfit files'
    d_save = r'E:\water database\data - 2021-08\air water\labfit files'


d_vac = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\vacuum scans'
d_vac = r'E:\water database\data - 2021-08\vacuum scans'


Patm = 628 # Torr, atmospheric pressure during measurement

cutoff = 0.15 # transmission threshold under which we don't trust the measurement (eg 25% -> 0.25)
cutoff_locations = {}
wvn2_cutoff = [6600, 7700] # what wavenumbers to include when looking for saturated features 

molecule_id = 1 # hitran molecular number for water

nuLow = 0
nuHigh = 0
shift = 0.0000000 # here if you want it (not sure why you would)

bins = np.array([6500.0, 6562.8, 6579.7, 6599.5, 6620.6, 6639.4, 6660.2, 6680.1, 6699.6, 6717.9,
                 6740.4, 6761.0, 6779.6, 6801.8, 6822.3, 6838.3 ,6861.4, 6883.2, 6900.1, 6920.2,
                 6940.0, 6960.5, 6982.9, 7002.5, 7021.4, 7041.1, 7060.5, 7081.7, 7099.0, 7119.0, 
                 7141.4, 7158.3, 7177.4, 7198.2, 7217.1, 7238.9, 7258.4, 7279.7, 7301.2, 7321.2, 
                 7338.9, 7358.5, 7377.1, 7398.5, 7421.0, 7440.8, 7460.5, 7480.6, 7500.1, 7520.4,
                 7540.6, 7560.5, 7580.5, 7600.0, 7620.0, 7640.0, 7660.0, 7720.0, 7800.0])

bins_count = np.ones((len(bins)-1, len(d_base)))

inp_file_conditions = []

for which_file in range(len(d_base)): # check with d_base[which_file]
    
    # which_file = 6 #####################################################################################
    # print('\n\n\n\n\n\n\n ********************\n hard coded file alert\n ********************\n\n\n\n\n\n\n\n\n\n')

    file_number = int(np.ceil((file_number+.1)/100) * 100)
    cutoff_locations[d_base[which_file]] = []
        
# %% load the files and data    
    d_load = os.path.join(d_meas, d_base[which_file] + ' bg subtraction.pckl')   
    
    f = open(d_load, 'rb')
    # what you get (first two): trans_BGremoved = (meas / bl) - H2O_bg, trans_BGincluded = (meas / bl) 
    [trans_BGremoved, trans_BGincluded, wavenumbers, T_meas, P_meas, y_h2o_meas, pathlength, favg, fitresults, model2020, _, _] = pickle.load(f)
    f.close() 
    
    if remove_bg: transmission = trans_BGremoved
    else: transmission = trans_BGincluded
    
    if remove_bg is False: # if you want to include this in labfit, we'll need the data
        
        if two_BG_temps: 
            print('\n\n********   this file is not ready for two background temperatures   ********')
            print('either it needs an upgrade, you need to use 1 background condition, or just remove BG before labfot')
            print('          sorry\n\n')
    
        d_load = os.path.join(d_vac, 'BL conditions.pckl')   
        f = open(d_load, 'rb')
        if d_ref: [bg_conditions, _] = pickle.load(f)
        else: [bg_conditions] = pickle.load(f)
        f.close() 

# %% prepare the conditions for labfit   

    #####################################################################################
    if d_type == 'pure': T = fitresults[4] # fit temperature using HITRAN 2020
    elif d_type == 'air': T = T_meas
    #####################################################################################
    
    P = P_meas
    
    T = T - 273.15 # in C
    L = pathlength / 100 # in m
    yh2o = y_h2o_meas
    
    # %% load the background conditions (if you're not subtracting out background before going to labfit)
    
    if remove_bg is False: 
        y_h2o_BG_fit = bg_conditions[which_BG[which_file],0]
        P_BG_fit = bg_conditions[which_BG[which_file],2] 
        
        y_h2o_BG = P_BG_fit * y_h2o_BG_fit / Patm # correct concentration to reflect the actual pressure in the room 

        T_BG = bg_conditions[which_BG[which_file],4] - 273.15 # in C
        L_BG = bg_conditions[which_BG[which_file],8] / 100 # in m
        
# %% remove points below given transmission level (combs are struggling to resolve low transmission features)
    
    # [istart, istop] = td.bandwidth_select_td(wavenumbers, wvn2_cutoff, max_prime_factor=False, print_value=False)

    plot_residual_offset = 1.05
    fit_order = 300
    fit_order_simple = 20
    fit_cutoff = 0.999 # fit the baseline (not transmission)
    
    # edge case that gave weird discontinuity
    if d_base[which_file] in ['300 K 600 T', '500 K 600 T'] and d_type == 'air': 
        fit_order = 300
        fit_cutoff = 0.995 # fit the baseline (not transmission)
    
    bl_fit = np.polynomial.chebyshev.Chebyshev.fit(wavenumbers[model2020 > fit_cutoff], transmission[model2020 > fit_cutoff], fit_order)      
    bl = bl_fit(wavenumbers) # improved baseline correction for entire spectrum 
 
    transmission_bl = transmission / bl
  
    bl_fit_simple = np.polynomial.chebyshev.Chebyshev.fit(wavenumbers[model2020 > fit_cutoff], transmission[model2020 > fit_cutoff], fit_order_simple)
    bl_simple = bl_fit_simple(wavenumbers)
  
    print('\n\n************ applying a small ({}th order) Chebyshev correction to the whole range ************'.format(fit_order_simple))
    print('************ to improve Labfit Chebyshev convergence (otherwise would need unique inital guesses for each bin and file) ************\n\n')
    
    transmission = transmission / bl_simple
  
    if check_bins: 
        
        if d_type == 'air': line_type = '--'
        else: line_type = ''
        
        plt.figure(1)
        plt.plot(wavenumbers, transmission_bl, line_type)
        # plt.plot(wavenumbers, 2.02 - transmission_bl, line_type)
        # plt.plot(wavenumbers, model2020, line_type)
        # plt.plot(bins, np.ones_like(bins)+0.01, 'X', color='k',markersize=10)
    
    if not check_bins: 
        plt.figure()
        fig_keep = plt.gcf().number
        plt.title(d_base[which_file] +  ' saved portions')
        plt.xlabel('wavenumber (cm-1)')
        plt.ylabel('transmission')
        
        plt.hlines(cutoff, min(wavenumbers), max(wavenumbers), colors='k', linestyles='solid', linewidth=1)
        plt.hlines(plot_residual_offset, min(wavenumbers), max(wavenumbers), colors='k', linestyles='dashed', linewidth=1)
        
        
        plt.figure()
        fig_reject = plt.gcf().number
        plt.title(d_base[which_file] + ' rejected portions')
        plt.ylabel('transmission')
    
    i_stop_cutoff = 0
    counter = 0
    i_plot_start = 0
    
    while i_stop_cutoff < len(transmission):  # go until there aren't any features left
        
        # counter for how many files for this measurement (number of breaks made thus far)
        counter += 1 
        
        # first index on right side of feature where transmission rises above cutoff (first to keep)
        i_start_cutoff = np.argmax(transmission_bl[i_stop_cutoff:-1] > cutoff) + i_stop_cutoff 
        
        # first index on left side of feature where transmission is about to fall below the cutoff (first to reject)
        i_stop_cutoff = np.argmax(transmission_bl[i_start_cutoff:-1] < cutoff) + i_start_cutoff
        
        # first index on right side of feature where transmission rises above cutoff (next to keep)
        i_start_cutoff_next = np.argmax(transmission_bl[i_stop_cutoff:-1] > cutoff) + i_stop_cutoff 
        
        # find the next stopping point (let's not bother splitting the file for a single skipped datapoint)
        while i_start_cutoff_next == i_stop_cutoff + 1: 
            # shift the end point to the next cutoff crossing
            i_stop_cutoff = np.argmax(transmission_bl[i_start_cutoff_next:-1] < cutoff) + i_start_cutoff_next    
            i_start_cutoff_next = np.argmax(transmission_bl[i_stop_cutoff:-1] > cutoff) + i_stop_cutoff 
                        
        # ignore points too close together (we're not making a file with less than 50 data points in it, labfit usually flips out anyway)
        # note edge case where we're making the last file (exit loop and enter next if statement)
        while i_stop_cutoff - i_start_cutoff < 50 and i_stop_cutoff != i_start_cutoff_next:
            # shift start point to the far side of the next cutoff crossing and use that region (small region is never saved)
            i_start_cutoff = i_start_cutoff_next
            i_stop_cutoff = np.argmax(transmission_bl[i_start_cutoff:-1] < cutoff) + i_start_cutoff 
            i_start_cutoff_next = np.argmax(transmission_bl[i_stop_cutoff:-1] > cutoff) + i_stop_cutoff 
            
        if i_stop_cutoff == i_start_cutoff_next: # we've reached the end
            i_stop_cutoff = len(transmission)
               
        # wavenumbers_snip = wavenumbers[i_start_cutoff+15:i_stop_cutoff-15] ################################################################
        # transmission_snip = transmission[i_start_cutoff+15:i_stop_cutoff-15] ################################################################
        
        wavenumbers_snip = wavenumbers[i_start_cutoff:i_stop_cutoff] ################################################################
        transmission_snip = transmission[i_start_cutoff:i_stop_cutoff] ################################################################
        
        # plot what you're keeping
        
        if counter == 1: # add legend entries on the first iteration
            label_keep_meas = 'measured'
            label_keep_residual = 'meas-model+'+str(plot_residual_offset)
        else: # we don't need 50 legend entries (stop adding them)
            label_keep_meas = ''
            label_keep_residual = ''        
        
        if not check_bins: 
            plt.figure(fig_keep)
            plt.plot(wavenumbers_snip, 
                     transmission_snip, label=label_keep_meas)
            plt.plot(wavenumbers[i_start_cutoff:i_stop_cutoff], 
                     transmission_bl[i_start_cutoff:i_stop_cutoff] - model2020[i_start_cutoff:i_stop_cutoff] + plot_residual_offset, label=label_keep_residual)
            plt.legend(loc='upper right')
            
            # plot what you're rejecting
            plt.figure(fig_reject)
    
            i_plot_span = np.arange(i_start_cutoff_next - i_stop_cutoff + 2) + i_plot_start 
            i_plot_span_wide = np.arange(i_start_cutoff_next - i_stop_cutoff + 8) + i_plot_start - 3
            
            if len(i_plot_span) > 0: # if there are points left to plot
                
                plt.plot(i_plot_span_wide, 
                         model2020[i_stop_cutoff-4:i_start_cutoff_next+4], '--', color='gray') # model + 1 point on each side
                plt.plot(i_plot_span_wide, 
                         transmission_bl[i_stop_cutoff-4:i_start_cutoff_next+4], 'k:') # transmission + 1 point on each side
                plt.plot(i_plot_span_wide, 
                          transmission_bl[i_stop_cutoff-4:i_start_cutoff_next+4] - model2020[i_stop_cutoff-4:i_start_cutoff_next+4] + 1-plot_residual_offset, 'k:') # residual + 1 point on each side
                
                plt.plot(i_plot_span, 
                         transmission_bl[i_stop_cutoff-1:i_start_cutoff_next+1]) # transmission removed
                plt.plot(i_plot_span, 
                         transmission_bl[i_stop_cutoff-1:i_start_cutoff_next+1] - model2020[i_stop_cutoff-1:i_start_cutoff_next+1] + 1-plot_residual_offset) # residual removed
        
                plt.hlines(cutoff, i_plot_span_wide[0], i_plot_span_wide[-1], colors='k', linestyles='solid', linewidth=1)
                plt.hlines(1-plot_residual_offset, i_plot_span_wide[0], i_plot_span_wide[-1], colors='k', linestyles='dashed', linewidth=1)
    
                i_plot_start = i_plot_span_wide[-1] + 3
          
        if wavenumbers_snip[-1] < wavenumbers[-1]: 
            bins_count[np.argmax(bins>wavenumbers_snip[-1])-1, which_file] += 1 # there is a bin break in this bin for this file
         
            
# %% process and save ASC file for labfit   
    
        file_number +=1 
        labfitname = str(file_number).zfill(4)
        
        # if file_number in [2820, 2825, 6702]:  # remove a point from the end. Not sure why this is necesary, but labfit will poop itself if you don't for these files
        #     wavenumbers_snip = wavenumbers_snip[0:-1]
        #     transmission_snip = transmission_snip[0:-1]
        
        descriptor = 'pure water measurement in CU low pressure furnace at ' + d_base[which_file] 
        
        '''
        Constructs Labfit input .asc file
        
        dataarray: input array of reduced comb data
        labfitname: 4 digit labfit identifier (str)
        descriptor: Description (str)
        T: temperature (C)
        P: pressure (torr)
        L: pathlength (m)
        yh2o: molefraction
        nuLow: low range of frequencies (cm-1), set to 0 for no filtering, otherwise use nuLow and nuHigh to define a subset of the spectrum in dataarray that is passed to the .asc file
        nuHigh: high range of frequencies
        mnum: molecule ID (hitran)
        
        '''
        
        np.set_printoptions(15)
                
        # format main values for labfit file
        Lstr = '%.7f' % L
        Tstr = ('%.4f' % T).rjust(12)
        Pstr = ('%.5f' % P).rjust(13)
        ystr = '%.6f' % yh2o

        if remove_bg is False:            
            
            Lstr_BG = ('%.7f' % L_BG).rjust(13)
            Tstr_BG = ('%.4f' % T_BG).rjust(12)
            Pstr_BG = ('%.5f' % Patm).rjust(13)
            ystr_BG = ('%.9f' % y_h2o_BG).ljust(12) # %.9 is most it will support, %.6 is most it will carry between iterations (still exploring what that means)

            
        delta = favg / speed_of_light / 100 # speed of light in cm-1
        delta = '%.30f' %  delta     #cm-1 ; note notation suppresses scientific notation
        
        wavenumbers_snip -= shift
        
        transmission_snip = 100*transmission_snip # I had it scaled 0-1 (labfit wants 0-100)
        
        if nuLow != 0:
            mask = (wavenumbers_snip > nuLow) & (wavenumbers_snip < nuHigh)
            wavenumbers_snip = np.extract(mask,wavenumbers_snip)
            transmission_snip = np.extract(mask,transmission_snip)
        
        startwn = '%.10f' % wavenumbers_snip[0]
        endwn = '%.10f' % wavenumbers_snip[-1]
        startwn_other = '%.10f' % wavenumbers_snip[0]
        
        fname = labfitname + "_" + d_base[which_file].replace("_", "-").replace(" ", "_") + "_" + str(counter) + ".asc"
        
        if save_data: 
            
            file = open(os.path.join(d_save,fname),'w')
            file.write("******* File "+labfitname+", "+descriptor+"\n")
            file.write(labfitname + "   " + startwn + "   " + endwn + "   " + delta+"\n")
            file.write("  00000.00    0.00000     0.00e0     " + str(molecule_id) + "   2     3   0        0\n")
            file.write("    " + Lstr + Tstr + Pstr + "    " + ystr + "    .0000000 .0000000 0.000\n")
            
            if remove_bg: file.write("    0.0000000     23.4486      0.00000    0.000000    .0000000 .0000000 0.000\n") # nothing
            else: file.write(Lstr_BG + Tstr_BG + Pstr_BG + "    " + ystr_BG + ".0000000 .0000000 0.000\n") # background conditions
            
            file.write("    0.0000000     23.4486      0.00000    0.000000    .0000000 .0000000 0.000\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n")
            file.write("DATE " + time.strftime("%m/%d/%Y") + "; time " + time.strftime("%H:%M:%S") + "\n")
            file.write("\n")
            # this line is an artifact of Ryan Cole. I'm not sure what the hard-coded numbers mean - scott
            file.write(startwn_other + " " + delta + " 15031 1 1    2  0.9935  0.000   0.000 294.300 295.400 295.300   7.000  22.000 500.000 START\n")
            
            wavelist = wavenumbers_snip.tolist()
            translist = transmission_snip.tolist()
            
            for i in range(len(wavelist)):
                wavelist[i] = '%.10f' %  wavelist[i]
                translist[i] = '%.10f' % translist[i]
                file.write(wavelist[i] + "      " + translist[i] + "\n")
            file.close()
            
            print("Labfit Input Generated for Labfit file " + labfitname)
            # these values are also in the inp file (which I think labfit prefers to use) You will want to change them there to match the ASC file (if needed)
            print(str(counter) + '          ' + delta[:8] + '       ' + str(T + 273.15).split('.')[0] + '     ' + Pstr[:-4] + '         ' + ystr[:5] + '\n') 
        
            cutoff_locations[d_base[which_file]].append([fname, np.round(wavenumbers_snip[0],3), np.round(wavenumbers_snip[-1],3)])

    #%% prep conditions so you can copy and paste them into a generic INP file
    
        if counter == 1: 
            
            Tstr = ('%.7f' % T)
            Pstr = ('%.7f' % P)
            delta = delta[:8] # checked that it was the same for all of my files (not included)
        
            inp_file_conditions.append([fname, Tstr, Pstr])
    
    #%% save the boundaries between files for use later
    
if save_data: 
    f = open(os.path.join(d_save, 'cutoff locations ' + d_type + '.pckl'), 'wb')
    pickle.dump(cutoff_locations,f)
    f.close() 

