# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:57:23 2020

@author: scott

calculate the average conditions, assuming pressure varies linearly during the measurement
(conservative estimate since it really varies exponentially and flattens out a lot after a few minutes)

"""

import numpy as np
import os
import pickle

Pstart = [19.900, 40.100, 59.450, 79.800, 119.550, 159.650, 320.150, 600.150,
      39.800, 79.300, 162.200, 321.200, 601.700,
      39.800 ,79.850, 160.100, 320.100, 600.500,
      39.900, 80.000, 159.800, 319.500, 600.400,
      39.850, 80.000, 159.900, 320.000, 600.150, 600.100]

Pend = [19.900, 40.050, 59.450, 79.800, 119.500, 159.650, 320.150, 600.100, 
      39.800, 79.500, 162.000, 320.900, 601.600, 
      39.800, 79.850, 160.150, 320.100, 600.700, 
      39.900, 80.050, 159.800, 319.500, 600.500, 
      39.950, 80.050, 159.950, 320.000, 600.100, 600.100]

t = [60]

for i in range(len(Pstart)):
    
    print(i)
    
    Ptime = np.linspace(Pstart[i], Pend[i], t[0])
    
    if i == 0: 
    
        Pavg = np.average(Ptime)
        Pstd = np.std(Ptime)
        Prel = 100 * Pstd / Pavg
    
    else: 
        
        Pavg = np.append(Pavg, np.average(Ptime))
        Pstd = np.append(Pstd, np.std(Ptime))
        Prel = np.append(Prel, 100 * np.std(Ptime) / np.average(Ptime))
    
#    print('Average Pressure =', round(Pavg,5), '+/-', round(Pstd,5), 'for a relative variation of', round(Prel,1), '%')


please = stophere

#%% calculated conditions for all measurements (save them for use later)

P_pure = [0.543131164, 1.034902538, 1.516729091, 2.022920455, 3.009446648, 4.000448011, 7.973403801, 15.91384573, 
          1.072195615, 2.031373552, 3.997464565, 7.954011401, 15.87903886, 
          1.031421851, 2.016456321, 4.025807303, 7.968431391, 15.8492044, 
          1.023217374, 2.000047367, 4.088956913, 7.95351416, 15.91384573, 
          1.014018415, 2.002036332, 3.987022503, 8.00224378 ,15.91384573, 
          16.06301804] # yes corrected using calibration data

P_air = [20.1565034, 40.2999598, 59.6446669, 79.9628496, 119.6257395, 159.6880040, 319.9370619, 599.4743203, 
         40.0253897, 79.5634750, 162.1341734, 320.8356547, 600.9969360, 
         40.0253897, 80.0127715, 160.1622613, 319.8871400, 599.9485776,
         40.1252334, 80.1874978 ,159.8377694 ,319.2880781 ,599.7988122,
         40.1252334, 80.1874978, 159.9625740, 319.7872964, 599.4743203,
         599.4493594] # yes corrected using calibration data

T_pure = [294.963, 295.088, 295.125, 295.100, 295.200, 295.250, 295.325, 295.500, 
          504.563, 504.375, 504.338, 504.588, 504.938, 
          703.850, 703.238, 703.588, 703.325, 703.850, 
          900.813, 901.088, 900.663, 900.763, 900.775,
          1099.163, 1099.613, 1099.563, 1099.113, 1099.050,
          1287.738] # not corrected using temperature logger

T_air = [295.300, 295.275, 295.200, 295.288, 295.200, 295.213, 295.175, 295.100, 
         504.850, 504.725, 504.450, 505.713, 505.975,
         703.600, 703.600, 704.175, 704.150, 703.463,
         900.913, 900.863, 900.788, 900.763, 900.950,
         1099.400, 1099.125, 1099.263, 1099.400, 1099.488, 
         1287.538] # not corrected using temperature logger

d_folder = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\air water'

d_final = os.path.join(d_folder, 'Air Water P & T.pckl')

f = open(d_final, 'wb')
pickle.dump([P_air, T_air], f)
f.close() 



