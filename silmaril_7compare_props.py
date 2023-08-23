

import numpy as np
import pickle 
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')


import clipboard_and_style_sheet
clipboard_and_style_sheet.style_sheet()





d_meas = r'C:\Users\scott\Documents\1-WorkStuff\High Temperature Water Data\data - 2021-08\pure water'







d_load = os.path.join(d_meas, 'output fit results - copy.pckl')

f = open(d_load, 'rb')
[T_all, P_all, y_h2o, pathlength, output2020, output2016, outputPaul, outputSceg] = pickle.load(f)
f.close()
    

output_all = [output2020, output2016, outputPaul, outputSceg]
output_names = ['output2020', 'output2016', 'outputPaul', 'outputSceg']
markers = ['x','3','+','o']

#%%

plt.figure()


for i, output in enumerate(output_all): 
    
    # plt.plot(T_all, output[:,4], 'x', label=output_names[i])

    plt.plot(T_all, 100*(T_all - output[:,4]) / T_all, marker=markers[i], linestyle = 'none', label=output_names[i])

    print('{}     {} ± {}'.format(output_names[i], np.mean(100*(T_all - output[:,4]) / T_all), np.std(100*(T_all - output[:,4]) / T_all)))


plt.legend()


#%%

plt.figure()

for i, output in enumerate(output_all): 
    
    # plt.plot(P_all, output[:,4], 'x', label=output_names[i])

    plt.plot(P_all, 100*(P_all - output[:,2]) / P_all, marker=markers[i], linestyle = 'none', label=output_names[i])

    print('{}     {} ± {}'.format(output_names[i], np.mean(100*(P_all - output[:,2]) / P_all), np.std(100*(P_all - output[:,2]) / P_all)))

plt.legend()


#%%

plt.figure()

for i, output in enumerate(output_all): 
    
    plt.plot(T_all, output[:,6], marker=markers[i], linestyle = 'none', label=output_names[i])
    
    print('{}     {} ± {}'.format(output_names[i], np.mean(output[:,6]), np.std(output[:,6])))


plt.legend()









