
import numpy as np
import matplotlib.pyplot as plt

import os
from sys import path
path.append(os.path.abspath('..')+'\\modules')

import linelist_conversions as db
import labfithelp as lab

path = r"C:\Users\scott\Documents\1-WorkStuff\code\Silmaril-to-LabFit-Processing-Scripts\data - HITRAN 2020\H2O.par"

hitran = np.genfromtxt(path, delimiter='\n', dtype='str')

df = db.par_to_df(path)

df['uncNU'] = df['ierr'].str[0].astype(int)
df['uncS'] = df['ierr'].str[1].astype(int)



cutoff_s296 = 1E-22
base = 10

for T_i in [300, 1100]: 
    
    cutoff_strength_atT = lab.strength_T(T_i, df.elower, df.nu) * cutoff_s296
    cutoff_strength = base**(2*np.log(cutoff_s296)/np.log(base) - np.log(cutoff_strength_atT)/np.log(base)) # reflect across S296
       
    df['ratio_'+str(T_i)] = np.log((df.sw / cutoff_strength) * (296 / T_i)) / np.log(base) # ratio of strength and cuttoff and ideal gas estimate for # molecules (at fixed P and V)

which_bool = (df.uncNU > 3) & (df.uncS == 6) & (df.ratio_300 > 0) & (df.ratio_1100 > 0)

df_keep = df[which_bool]

df_keep['source_S'] = df_keep.iref.str[2:4].astype(int)



#%%

unc_cutoff = 5

# plt.plot(df.nu, 1 + df.ratio_300, 'yx', markersize=5)
# plt.plot(df.nu, 1 + df.ratio_1100, 'cx', markersize=5)

# plt.plot(df_keep[df_keep.uncNU>5].nu, 1 + df_keep[df_keep.uncNU>5].ratio_300, 'yX', markersize=10)
# plt.plot(df_keep[df_keep.uncNU>5].nu, 1 + df_keep[df_keep.uncNU>5].ratio_1100, 'cX', markersize=10)
# plt.plot(df_keep[df_keep.uncNU>5].nu, 1 + df_keep[df_keep.uncNU>5].ratio_1100/1e15, 'kX', markersize=10)

    
# plt.plot(df_keep[df_keep.uncNU<=5].nu, 1 + df_keep[df_keep.uncNU<=5].ratio_300, 'yo', markersize=10)
# plt.plot(df_keep[df_keep.uncNU<=5].nu, 1 + df_keep[df_keep.uncNU<=5].ratio_1100, 'co', markersize=10)
# plt.plot(df_keep[df_keep.uncNU<=5].nu, 1 + df_keep[df_keep.uncNU<=5].ratio_1100/1e15, 'ko', markersize=10)



#%%

span = [[7010.2,7010.7],[7012.45,7012.95],[7026.35,7027],[7038.1,7038.8],[7041.8,7042.8],
        [7046.9,7048.3],[7070.1,7071.2],[7071.05,7072.3],[7085.5,7086.3],[7093.7,7095.3],[7104.3,7105.3],
        [7116.6,7118.6],[7138.4,7140.4],[7160.95,7162.1],[7167.6,7169.6],[7193.6,7196],[7198.5,7200],
        [7201.5,7203.5],[7204.4,7206.2],[7212.3,7213.4],[7214.9,7215.9],[7215.9,7216.9],[7218.6,7220.3],
        [7241.55,7245],[7293.05,7295.6],[7311.4,7313.5],[7326.9,7329.3],[7338.5,7341.4],[7347.6,7349.2],
        [7355, 7356.6],[7374.1,7376.17],[7389,7390.6],[7396.7,7398.5],[7405.5,7406.3],[7412.8,7413.4],
        [7413.3,7414.4],[7415.8,7416.6],[7417.2,7418.3]]




# plt.plot(df.nu, 1 + df.ratio_300, 'yx', markersize=5)
# plt.plot(df.nu, 1 + df.ratio_1100, 'cx', markersize=5)

# plt.plot(df_keep.nu, 1 + df_keep.ratio_300, 'yo', markersize=10)
# plt.plot(df_keep.nu, 1 + df_keep.ratio_1100, 'co', markersize=10)
# plt.plot(df_keep.nu, 1 + df_keep.ratio_1100/1e15, 'ko', markersize=10)

for features in span: 
    
    plt.plot(features, [1.009,1.011], 'k')
    plt.plot(features[0], 1.01, 'kX', markersize=15)
    plt.plot(features[1], 1.01, 'kX', markersize=15)


# wvn2_concentration = [[7184.85, 7186.48], [7412.82, 7413.35], [7415.83, 7416.18], [6919.71, 6920.20], [6807.60, 6807.94], # Goldenstein (higher temp)
#                       [7134.30, 7135.25], [7135.70, 7136.65], [7138.50, 7139.39], [7139.39, 7139.97], [7138.50, 7139.97]] # other hanson features, lower temp


# for features in wvn2_concentration: 
    
#     plt.plot(features, [1.009,1.011], 'r')
#     plt.plot(features[0], 1.01, 'rX', markersize=15)
#     plt.plot(features[1], 1.01, 'rX', markersize=15)

plt.xlabel('wavenumber')


#%% find features in each range

feature_info = {}

for i_span, features in enumerate(span): 
    
    df_keep_span = df_keep[(df_keep.nu>features[0]) & (df_keep.nu<features[1])]
    
    feature_info[i_span] = df_keep_span[['nu','sw','elower','quanta','ratio_300','ratio_1100','uncNU','uncS']].to_numpy()
    
    print(df_keep_span.source_S)
    

















