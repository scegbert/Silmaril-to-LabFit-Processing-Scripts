
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle 
import os
from math import inf
import pandas as pd

d_meas = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\new data\pure water'
d_save = r'C:\Users\scott\Documents\1-WorkStuff\water campaign\new data\pure water\labfit files'

d_base = ['300 K _5 T', '300 K 1 T', '300 K 1_5 T', '300 K 2 T', '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
          '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
          '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
          '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
          '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T']

d_base = ['300 K 16 T']
   
yh2o = 1
mnum = 1 # hitran molecular number for water

nuLow = 0
nuHigh = 0
shift = 0.0000000 # here if you want it (not sure why you would)

for which_file in range(len(d_base)): # check with d_base[which_file]

# %% load the files and data    
    d_load = os.path.join(d_meas, d_base[which_file] + ' bg subtraction.pckl')
    
    f = open(d_load, 'rb')
    [transmission, wavenumbers, T_meas, P_meas, pathlength, favg, fitresults, model, _, _] = pickle.load(f)
    f.close() 
    
    P_fit2020 = fitresults[0]
    T_fit2020 = fitresults[2]

    T = T_fit2020
    P_cal = [0.0, 0.99448205, 0.02352257]
    P = P_meas**2 * P_cal[0] + P_meas * P_cal[1] + P_cal[2]  # in torr, corrected using most recent calibration
    
    T = T - 273.15 # in C
    L = pathlength / 100 # in m        
        
# %% process and save ASC file for labfit


    labfitname = '1' + str(which_file).zfill(3)
    
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
    
    ### Constants
    c = 29979245800 #cm/s
    
    molfrac1 = yh2o
    molec_id1 = mnum
    
    ### Write labfit files
    molfrac1 = '%.6f' % molfrac1
    
    Lstr = '%.7f' % L
    Tstr = ('%.4f' % T).rjust(12)
    Pstr = ('%.5f' % P).rjust(13)
    
    delta = favg / c
    delta = '%.31f' %  delta     #cm-1 ; note notation suppresses scientific notation
    
    wavenumbers -= shift
    
    transmission = 100*transmission # I had it scaled 0-1 (labfit wants 0-100)
    
    if nuLow != 0:
        mask = (wavenumbers > nuLow) & (wavenumbers < nuHigh)
        wavenumbers = np.extract(mask,wavenumbers)
        transmission = np.extract(mask,transmission)
    """
    plt.plot(wavenumbers,transmission)
    bins = [6600.0, 6619.0, 6640.0, 6660.0, 6680.0, 6699.4, 6718.0, 6739.6, 6757.2, 6780.0, 6802.2,
            6823.0, 6838.3, 6861.1, 6884.2, 6899.5, 6921.3, 6938.5, 6959.4, 6982.5, 6996.7, 7020.9,
            7041.0, 7060.7, 7077.5, 7098.7, 7121.0, 7141.7, 7158.6, 7176.7, 7200.7, 7216.8, 7238.3,
            7258.0, 7279.5, 7300.7, 7320.0, 7338.8, 7358.5, 7376.9, 7398.5, 7422.0, 7440.7, 7460.6,
            7481.0, 7500.0, 7520.0, 7540.0, 7560.5, 7580.7, 7600.0]
    for i in range(len(bins)):
        plt.axvline(x=bins[i], color='k', linestyle='--')
    plt.plot(wvn, model_trans)
    """

    
    startwn = '%.10f' % wavenumbers[0]
    endwn = '%.8f' % wavenumbers[-1]
    startwn_other = '%.7f' % wavenumbers[0]
    
    fname = labfitname + "_" + d_base[which_file].replace("_", "-").replace(" ", "_") + ".asc"
    
    file =open(fname,'w')
    file.write("******* File "+labfitname+", "+descriptor+"\n")
    file.write(labfitname + "   " + startwn + "   " + endwn + "   " + delta+"\n")
    file.write("  00000.00    0.00000     0.00e0     " + str(molec_id1) + "   2     3   0        0\n")
    file.write("    " + Lstr + Tstr + Pstr + "    " + molfrac1 + "    .0000000 .0000000 0.000\n")
    file.write("    0.0000000     23.4486      0.00000    0.000000    .0000000 .0000000 0.000\n")
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
    file.write(startwn_other + " " + delta + " 15031 1 1    2  0.9935  0.000   0.000 294.300 295.400 295.300   7.000  22.000 500.000 START\n")
    
    
    wavelist = wavenumbers.tolist()
    translist = transmission.tolist()
    
    for i in range(len(wavelist)):
        wavelist[i] = '%.5f' %  wavelist[i]
        translist[i] = '%.5f' % translist[i]
        file.write(wavelist[i] + "      " + translist[i] + "\n")
    file.close()
    
    print("Labfit Input Generated for Labfit file " + labfitname)
    print(delta[0:8] + '   ' + Tstr + '   ' + Pstr) # these values are also in the inp file (which I think labfit prefers to use) You might want to change them
    print()
    


