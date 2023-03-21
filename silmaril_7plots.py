
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from sys import path
path.append(r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\modules')

import linelist_conversions as db
import fig_format
from hapi import partitionSum # hapi has Q(T) built into the script, with this function to call it

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import labfithelp as lab

import os
import pickle
import pldspectrapy as pld
import td_support as td


# %% define some dictionaries and parameters

d_type = 'pure' # 'pure' or 'air'

if d_type == 'pure': 
    d_conditions = ['300 K _5 T', '300 K 1 T', '300 K 1_5 T', '300 K 2 T', '300 K 3 T', '300 K 4 T', '300 K 8 T', '300 K 16 T', 
                    '500 K 1 T',  '500 K 2 T',  '500 K 4 T',  '500 K 8 T',  '500 K 16 T', 
                    '700 K 1 T',  '700 K 2 T',  '700 K 4 T',  '700 K 8 T',  '700 K 16 T', 
                    '900 K 1 T',  '900 K 2 T',  '900 K 4 T',  '900 K 8 T',  '900 K 16 T', 
                    '1100 K 1 T', '1100 K 2 T', '1100 K 4 T', '1100 K 8 T', '1100 K 16 T']
elif d_type == 'air': 
    d_conditions = ['300 K 20 T', '300 K 40 T',  '300 K 60 T','300 K 80 T',  '300 K 120 T', '300 K 160 T', '300 K 320 T', '300 K 600 T', 
                    '500 K 40 T',  '500 K 80 T',  '500 K 160 T',  '500 K 320 T',  '500 K 600 T', 
                    '700 K 40 T',  '700 K 80 T',  '700 K 160 T',  '700 K 320 T',  '700 K 600 T', 
                    '900 K 40 T',  '900 K 80 T',  '900 K 160 T',  '900 K 320 T',  '900 K 600 T', 
                    '1100 K 40 T', '1100 K 80 T', '1100 K 160 T', '1100 K 320 T', '1100 K 600 T']

props = {}
props['nu'] = ['nu', 'ν', 1, 23, 0.0015] # name in df, symbol for plotting, location of float (0 or 1) in INP file, number for constraint, acceptable uncertainty for fitting
props['sw'] = ['sw', '$S_{296}$', 2, 24, 0.09] # 9 % percent
props['gamma_air'] = ['gamma_air', 'γ air', 3, 25, 0.10] # ? (might be too generous for gamma only fits)
props['elower'] = ['elower', 'E\"', 4, 34, None]
props['n_air'] = ['n_air', 'n air', 5, 26, 0.13]
props['delta_air'] = ['delta_air', 'δ air', 6, 27, .005]
props['n_delta_air'] = ['n_delta_air', 'n δ air', 7, 28, 0.2]
props['MW'] = ['MW', 'MW', 8, 29, None]
props['gamma_self'] = ['gamma_self', 'γ self', 9, 30, 0.10]
props['n_self'] = ['n_self', 'n γ self', 10, 31, 0.13]
props['delta_self'] = ['delta_self', 'δ self', 11, 32, 0.005]
props['n_delta_self'] = ['n_delta_self', 'n δ self', 12, 33, 0.13]
props['beta_g_self'] = ['beta_g_self', 'βg self', 13, 35, None] # dicke narrowing (don't worry about it for water)
props['y_self'] = ['y_self', 'y self', 14, 36, None] # rosenkrantz line mixing (don't worry about this one either)
props['sd_self'] = ['sd_self', 'speed dependence', 15, 37, 0.10] # pure and air
props[False] = False # used with props_which2 option

buffer = 2 # I added a buffer to avoid weird chebyshev edge effects
bin_breaks = [6656.0, 6680.0, 6699.3, 6718.1, 6739.6, 6757.1, 6779.6, 6802.2, 6823.0, 6838.3,
              6861.1, 6884.2, 6899.5, 6921.2, 6938.5, 6959.5, 6982.5, 6996.7, 7021.3, 7041.0, 7060.7, 7077.5, 7098.7, 
              7121.0, 7141.7, 7158.6, 7176.7, 7200.7, 7216.8, 7238.3, 7258.0, 7279.5, 7300.7, 7320.0, 7338.8, 7358.5, 
              7376.9, 7398.5, 7422.0, 7440.7, 7460.6, 7480.8, 7500.0, 7520.0, 7540.0]
bin_names = ['B1',  'B2',  'B3',  'B4',  'B5',  'B6',  'B7',  'B8',  'B9',  'B10', 'B11', 'B12', 'B13', 
            'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26',
            'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'B33', 'B34', 'B35', 'B36', 'B37', 'B38', 'B39',
            'B40', 'B41', 'B42', 'B43', 'B44']
if d_type == 'air': bin_names = [item + 'a' for item in bin_names] # append an a to the end of the bin name

bins = {} # dictionary (key is bin_names, entries are bin_breaks on either side)
for i in range(len(bin_names)): bins[bin_names[i]] = [-buffer, bin_breaks[i], bin_breaks[i+1], buffer] 

d_labfit_kp = r'C:\Users\scott\Documents\1-WorkStuff\Labfit - Kiddie Pool'
d_labfit_real = r'C:\Users\scott\Documents\1-WorkStuff\Labfit'

# lab.nself_quantumJpp(d_labfit_kp, base_name, d_labfit_kp) # change n_gamma_self to match paul's J" analysis
# lab.compile_results(d_labfit_real, base_name_air, base_name_air + '1', bins) # compile results into a single inp file (pure -> air)

base_name_pure = 'B2020Jp'
base_name_air = 'B2020Ja1'

if d_type == 'pure': base_name = base_name_pure
elif d_type == 'air': base_name = base_name_air

cutoff_s296 = 3E-24/3 # 3E-24/3 # ballpark NF for pure water
ratio_min = 0.3 # minimum S296 ratio to consider
offset = 4 # for plotting

if d_type == 'pure': props_which = ['nu','sw','gamma_self','n_self','sd_self','delta_self','n_delta_self', 'elower']
elif d_type == 'air': props_which = ['nu','sw','gamma_air','n_air','sd_self','delta_air','n_delta_air', 'elower']



# %% other stuff to put at the top here


colors = ['dodgerblue', 'firebrick', 'darkorange', 'darkgreen', 'purple', 'moccasin']
colors_grad = ['firebrick','orangered','goldenrod','forestgreen','teal','royalblue','mediumpurple', 'darkmagenta']


features_clean = [3426, 4018,                                                                                                                   # B1
                 4164, 4174, 4188, 4359, 4513,                                                                                                  # B2
                 4792, 4869, 4882, 4891,                                                                                                        # B3
                 5426, 5428, 5491, 5530, 5603, 5623, 5692, 5793, 5951, 5975,                                                                    # B4
                 6158, 6186, 6258, 6316, 6332, 6339, 6414, 6421, 6560, 6594, 6632, 6647,                                                        # B5
                 6729, 6753, 6764, 6777, 6806, 6959, 7090, 7091, 7111, 7117, 7141, 7148, 7195, 7233, 7261, 7300, 7302, 7348,                    # B6
                 7457, 7577, 7686, 7768, 7795, 7807, 7897, 7911, 7969, 7995, 8000, 8041, 8075, 8088, 8103, 8120, 8121, 8135,                    # B7
                 8247, 8248, 8260, 8312, 8318, 8375, 8417, 8427, 8456, 8481, 8597, 8659, 8674, 8678, 8739, 8777, 8783, 8849, 8859,              # B8
                 8965, 8979, 8986, 9068, 9193, 9288, 9299, 9336, 9384, 9399, 9428, 9429, 9443, 9457,                                            # B9
                 9492, 9518, 9560, 9565, 9572, 9601, 9603, 9606, 9633, 9685, 9730, 9756, 9759, 9760, 9786, 9836, 9862, 10225, 10230, 10242,     # B10
                 10296, 10297, 10328, 10348, 10349, 10409, 10410, 10436, 10464, 10509, 10514, 10535, 10556, 10587, 10632, 10709, 10760, 
                 10764, 10770, 10894, 10972, 11018,                                                                                             # B11
                 11119, 11155, 11215, 11240, 11345, 11379, 11392, 11401, 11403, 11424, 11430, 11431, 11441, 11448, 11451, 11460, 11508, 
                 11529, 11531, 11555, 11581, 11622, 11629, 11661,                                                                               # B12
                 11794, 11796, 11891, 11956, 12033, 12058, 12097, 12142, 12154, 12233, 12283, 12330, 12332, 12356, 12375, 12381, 12394,
                 12473, 12489, 12513, 12569, 12575,                                                                                             # B13
                 12601, 12635, 12677, 12680, 12687, 12733, 12736, 12813, 12945, 12968, 12982, 12984, 13018, 13056, 13100, 13133, 13173, 
                 13249, 13251,                                                                                                                  # B14
                 13441, 13455, 13467, 13537, 13544, 13561, 13586, 13602, 13606, 13624, 13637, 13644, 13653, 13655, 13663, 13664, 13673, 
                 13703, 13715, 13737, 13757, 13836, 13947, 13952, 13965, 13969, 13996, 14044, 14086, 14105, 14149,                              # B15
                 14231, 14234, 14253, 14321, 14350, 14426, 14492, 14495, 14496, 14558, 14656, 14669, 14671, 14678, 14691, 14718, 14719,
                 14726, 14732, 14768, 14804, 14807, 14824, 14835, 14925, 14976, 15027, 15033, 15060, 15067, 15081, 15103,                       # B16
                 15176, 15190, 15222, 15240, 15244, 15248, 15351, 15358, 15425, 15621, 15679,                                                   # B17
                 15786, 15799, 15804, 15809, 15813, 15848, 15873, 15892, 15932, 15937, 15963, 15966, 15970, 15977, 15984, 16054, 16079,
                 16170, 16186, 16253, 16266, 16365, 16392, 16464, 16474, 16485, 16538, 16539, 16597, 16604, 16612, 16722,                       # B18
                 16848, 16860, 16871, 16903, 16908, 16916, 16987, 17000, 17051, 17064, 17120, 17125, 17142, 17158, 17161, 17177, 17250, 
                 17263, 17272, 17303, 17360, 17394, 17413, 17483, 17564, 17582, 17593, 17600, 17610, 17623, 17643,                              # B19
                 17716, 17771, 17790, 17794, 17842, 17848, 17855, 17877, 17898, 17907, 17915, 17930, 17931, 17941, 17972, 18066, 18104,
                 18113, 18200, 18206, 18207, 18216, 18273, 18295, 18307, 18322, 18333, 18338, 18372, 18388, 18406,                              # B20
                 18520, 18584, 18604, 18624, 18638, 18671, 18688, 18733, 18784, 18791, 18815, 18820, 18823, 18828, 18857, 18862, 18937,
                 18972, 18973, 19043, 19058, 19180,                                                                                              # B21
                 19273, 19303, 19329, 19343, 19369, 19372, 19418, 19466, 19514, 19527, 19543, 19555, 19556, 19583, 19624, 19697, 19715,
                 19716, 19738, 19750, 19779, 19806, 19839, 19861, 19869, 19902, 19938, 19963, 19967, 19988, 19992, 20035, 20070, 20080, 
                 20094,                                                                                                                         # B22
                 20160, 20207, 20247, 20286, 20381, 20429, 20437, 20487, 20498, 20533, 20568, 20666, 20675, 20754, 20777, 20827, 20835, 
                 20845, 20874, 20880, 20930, 20963, 20974, 21043, 21070,                                                                        # B23
                 21119, 21128, 21153, 21162, 21167, 21215, 21227, 21318, 21364, 21372, 21402, 21463, 21535, 21579, 21630, 21671, 21693, 
                 21710, 21728, 21760, 21812, 21821, 21835, 21870, 21884, 21888, 21938, 21945,                                                   # B24
                 22009, 22017, 22023, 22042, 22120, 22125, 22364, 22387, 22426, 22439, 22452, 22527, 22528, 22634, 22638, 22649, 22676, 
                 22677,                                                                                                                         # B25
                 22750, 22793, 22812, 22835, 22947, 22996, 23018, 23024, 23053, 23078, 23117, 23216, 23315, 23367, 23446, 23464,                # B26
                 23578, 23656, 23664, 23689, 23723, 23746, 23759, 23853, 23854, 23959, 23996, 24008, 24015, 24022, 24047, 24062, 24138,
                 24207, 24222, 24232, 24237, 24244, 24443, 24447,                                                                               # B27
                 33710, 33766, 33808, 33863, 33960, 34068, 34112, 34219,                                                                        # B39
                 34260, 34268, 34363, 34401, 34441, 34544, 34628, 34634, 34635, 34658, 34838,                                                   # B40
                 34898, 34915, 35017, 35023, 35024, 35132, 35294, 35349,                                                                        # B41
                 35593]                                                                                                                         # B42
                 
features_doublets_grouped = [[3397, 3398],                                                                                                              # B1
                     [4133, 4134], [4147, 4148], [4328, 4329], [4384, 4385], [4411, 4412], [4438, 4439], [4488, 4489], [4653, 4654],            # B2
                     [4859, 4860], [4898, 4899], [5063, 5064], [5114, 5115], [5192, 5193], [5314, 5315],                                        # B3
                     [5438, 5439], [5468, 5469], [5495, 5496], [5671, 5672], [5893, 5894],                                                      # B4
                     [6131, 6132], [6262, 6263], [6305, 6306], [6319, 6320], [6389, 6390], [6393, 6394], [6400, 6401], [6464, 6465], 
                     [6488, 6489], [6519, 6520],                                                                                                # B5
                     [6721, 6722], [6819, 6820], [7175, 7176], [7189, 7190], [7221, 7222], [7242, 7243], [7283, 7284], [7319, 7320], 
                     [7321, 7322], [7371, 7372], [7382, 7383], [7391, 7392],                                                                    # B6
                     [7427, 7428], [7471, 7472], [7488, 7489], [7498, 7499], [7600, 7601], [7658, 7659], [7938, 7939], [7979, 7980], 
                     [7985, 7986], [8056, 8057], [8076, 8077], [8120, 8121], [8146, 8147],                                                      # B7
                     [8165, 8166], [8245, 8246], [8289, 8290], [8301, 8302], [8304, 8305], [8364, 8365], [8376, 8377], [8459, 8460], 
                     [8465, 8466], [8467, 8468], [8486, 8487], [8622, 8623], [8880, 8881],                                                      # B8
                     [9080, 9081], [9153, 9154], [9176, 9177], [9199, 9200], [9290, 9291], [9322, 9323], [9369, 9370], [9413, 9414], 
                     [9428, 9429],                                                                                                              # B9
                     [9488, 9489], [9502, 9503], [9533, 9534], [9594, 9595], [9642, 9643], [9759, 9760], [10063, 10064], [10131, 10132], 
                     [10152, 10153], [10248, 10249], [10256, 10257], [10261, 10262],                                                            # B10
                     [10296, 10297], [10304, 10305], [10348, 10349], [10389, 10390], [10409, 10410], [10448, 10449], [10504, 10505], 
                     [10921, 10922], [10962, 10963], [11086, 11087],                                                                            # B11
                     [11097, 11098], [11139, 11140], [11216, 11217], [11413, 11414], [11430, 11431], [11488, 11489], [11505, 11506], 
                     [11618, 11619], [11649, 11650],                                                                                            # B12
                     [11709, 11710], [11876, 11877], [11988, 11989], [12290, 12291], [12317, 12318], [12373, 12374], [12424, 12425], 
                     [12533, 12534], [12132, 12133],                                                                                            # B13
                     [12792, 12793], [13073, 13074], [13093, 13094], [13232, 13233], [13278, 13279],                                            # B14
                     [13371, 13372], [13558, 13559], [13691, 13692], [13821, 13822], [13883, 13884], [13982, 13983], [14005, 14006], 
                     [14056, 14057], [14096, 14097], [14130, 14131],                                                                            # B15
                     [14196, 14197], [14245, 14246], [14338, 14339], [14382, 14383], [14431, 14432], [14718, 14719], [14775, 14776], 
                     [15001, 15002],                                                                                                            # B16
                     [15213, 15214], [15251, 15252], [15275, 15276], [15564, 15565], [15577, 15578], [15592, 15593], [15619, 15620], 
                     [15626, 15627], [15655, 15656], [15697, 15698], [15707, 15708], [15752, 15753],                                            # B17
                     [15772, 15773], [15840, 15841], [15895, 15896], [15927, 15928], [15987, 15988], [16020, 16021], [16023, 16024], 
                     [16124, 16125], [16207, 16208], [16241, 16242], [16269, 16270], [16309, 16310], [16374, 16375], [16446, 16447], 
                     [16502, 16503], [16538, 16539], [16590, 16591], [16603, 16604], [16625, 16626], [16663, 16664], [16680, 16681], 
                     [16785, 16786], [16795, 16796], [16801, 16802], [16814, 16815],                                                            # B18
                     [16974, 16975], [17002, 17003], [17031, 17032], [17182, 17183], [17211, 17212], [17220, 17221], [17252, 17253], 
                     [17591, 17592], [17636, 17637],                                                                                            # B19
                     [17918, 17919], [17930, 17931], [18019, 18020], [18141, 18142], [18162, 18163], [18189, 18190], [18206, 18207], 
                     [18239, 18240], [18374, 18375],                                                                                            # B20
                     [18492, 18493], [18514, 18515], [18689, 18690], [18778, 18779], [18852, 18853], [18944, 18945], [18972, 18973], 
                     [18976, 18977], [19139, 19140], [19174, 19175],                                                                            # B21
                     [19232, 19233], [19322, 19323], [19334, 19335], [19381, 19382], [19461, 19462], [19539, 19540], [19555, 19556], 
                     [19647, 19648], [19715, 19716], [19825, 19826], [19999, 20000],                                                            # B22
                     [20154, 20155], [20177, 20178], [20205, 20206], [20273, 20274], [20306, 20307], [20351, 20352], [20478, 20479], 
                     [20544, 20545], [20582, 20583], [20636, 20637], [20767, 20768], [21023, 21024], [21081, 21082],                             # B23
                     [21146, 21147], [21170, 21171], [21261, 21262], [21442, 21443], [21455, 21456], [21478, 21479], [21480, 21481], 
                     [21489, 21490], [21519, 21520], [21802, 21803], [21809, 21810],                                                            # B24
                     [22131, 22132], [22213, 22214], [22272, 22273], [22291, 22292], [22355, 22356], [22427, 22428], [22432, 22433], 
                     [22500, 22501], [22527, 22528], [22676, 22677], [22681, 22682],                                                            # B25
                     [22979, 22981], [23275, 23276], [23353, 23354], [23459, 23460],                                                            # B26
                     [23626, 23629], [23630, 23631], [23681, 23682], [23853, 23854], [23966, 23967], [24002, 24003], [24057, 24058], 
                     [24334, 24335],                                                                                                            # B27
                     [33719, 33720], [33750, 33751], [33840, 33841], [33904, 33905], [33945, 33946], [34061, 34062], [34069, 34070],
                     [34076, 34077],                                                                                                            # B39
                     [34349, 34350], [34589, 34590], [34634, 34635], [34831, 34832],                                                            # B40
                     [35023, 35024], [35041, 35042], [35175, 35176], [35195, 35196], [35331, 35332], [35359, 35360],                            # B41
                     [35468, 35469], [35500, 35501], [35570, 35571], [35603, 35604], [35606, 35607], [35707, 35708], [35709, 35710],            # B42
                     [36160, 36161], [36514, 36515]]                                                                                            # B43
                             
features_doublets = [item for sublist in features_doublets_grouped for item in sublist]


# %% read in results

d_sceg = r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\silmaril pure water\data - sceg'

f = open(os.path.join(d_sceg,'df_sceg.pckl'), 'rb')
[df_sceg, df_og] = pickle.load(f)
f.close()

f = open(os.path.join(d_sceg,'spectra_air.pckl'), 'rb')
[T_air, P_air, wvn_air, trans_air, res_air, res_og_air] = pickle.load(f)
f.close()

f = open(os.path.join(d_sceg,'spectra_pure.pckl'), 'rb')
[T_pure, P_pure, wvn_pure, trans_pure, res_pure, res_og_pure] = pickle.load(f)
f.close()


q0offset = 0

quanta = df_sceg.quanta.str.replace('-',' -').replace('q','').str.split(expand=True).astype('int32') # looking for doublets (watch out for negative quanta without spaces, ie -2-2-2)
if quanta.isnull().to_numpy().any(): please = stop # if anything got input as None, that means we're missing something. Let's do a quick check for that  

df_sceg['v1p'] = quanta[0+q0offset].astype(int) # extract v1' value
df_sceg['v2p'] = quanta[1+q0offset].astype(int) # extract v2' value
df_sceg['v3p'] = quanta[2+q0offset].astype(int) # extract v3' value

df_sceg['v1pp'] = quanta[3+q0offset].astype(int) # extract v1" value
df_sceg['v2pp'] = quanta[4+q0offset].astype(int) # extract v2" value
df_sceg['v3pp'] = quanta[5+q0offset].astype(int) # extract v3" value

df_sceg['v_all'] = '('+df_sceg.v1p.astype(str)+df_sceg.v2p.astype(str)+df_sceg.v3p.astype(str)+')<-('+df_sceg.v1pp.astype(str)+df_sceg.v2pp.astype(str)+df_sceg.v3pp.astype(str)+')'

df_sceg['Jp'] = quanta[6+q0offset].astype(int) # extract J' value
df_sceg['Kap'] = quanta[7+q0offset].astype(int) # extract Ka' value
df_sceg['Kcp'] = quanta[8+q0offset].astype(int) # extract Kc' value

df_sceg['Jpp'] = quanta[9+q0offset].astype(int) # extract J" value
df_sceg['Kapp'] = quanta[10+q0offset].astype(int) # extract Ka" value
df_sceg['Kcpp'] = quanta[11+q0offset].astype(int) # extract Kc" value

df_sceg['r_all'] = '('+df_sceg.Jp.astype(str)+','+df_sceg.Kap.astype(str)+','+df_sceg.Kcp.astype(str)+')<-('+df_sceg.Jpp.astype(str)+','+df_sceg.Kapp.astype(str)+','+df_sceg.Kcpp.astype(str)+')' 

df_sceg['rv_all'] = '('+df_sceg.v1p.astype(str)+df_sceg.v2p.astype(str)+df_sceg.v3p.astype(str)+')<-('+df_sceg.v1pp.astype(str)+df_sceg.v2pp.astype(str)+df_sceg.v3pp.astype(str)+')+('+df_sceg.Jp.astype(str)+','+df_sceg.Kap.astype(str)+','+df_sceg.Kcp.astype(str)+')<-('+df_sceg.Jpp.astype(str)+','+df_sceg.Kapp.astype(str)+','+df_sceg.Kcpp.astype(str)+')' 

df_sceg['Jdelta'] = df_sceg['Jp'] - df_sceg['Jpp']
df_sceg['Kadelta'] = df_sceg['Kap'] - df_sceg['Kapp']
df_sceg['Kcdelta'] = df_sceg['Kcp'] - df_sceg['Kcpp']

df_sceg['taup_fam'] =  df_sceg['Kap'] +  df_sceg['Kcp']  - df_sceg['Jp'] # tau' family is 0 or 1
df_sceg['taupp_fam'] = df_sceg['Kapp'] + df_sceg['Kcpp'] - df_sceg['Jpp'] # tau'' family is 0 or 1

df_sceg['Jm'] = df_sceg[['Jp', 'Jpp']].max(axis=1) # max of J
df_sceg['Km'] = df_sceg[['Kap', 'Kapp']].max(axis=1) # max of Ka (see toth self-broadened widths)

df_sceg['m'] = 999
df_sceg.m[df_sceg.Jdelta == 1]  =  df_sceg.Jpp[df_sceg.Jdelta == 1] # m = J''+1 = J' for delta J = 1 (R branch)
df_sceg.m[df_sceg.Jdelta == 0]  =  df_sceg.Jpp[df_sceg.Jdelta == 0] # m = J'' when delta J = 0 (Q branch)
df_sceg.m[df_sceg.Jdelta == -1] = -df_sceg.Jpp[df_sceg.Jdelta == -1] # m = -J'' for delta J = -1 (P branch)


df_sceg['sw1100'] = lab.strength_T(1100, df_sceg.elower, df_sceg.nu) * df_sceg.sw
df_og['sw1100'] = lab.strength_T(1100, df_og.elower, df_og.nu) * df_og.sw

please = stophere


# %% quantum explorations

d_ht2020 = r'C:\Users\scott\Documents\1-WorkStuff\code\scottcode\silmaril pure water\data - HITRAN 2020\H2O.par'
df_ht2020 = db.par_to_df(d_ht2020) # currently shifted by one from df_sceg (labfit indexes from 1) 8181 HT <=> 8182 LF

# v_bands = ['(200)<-(000)', '(111)<-(010)', '(101)<-(000)', '(031)<-(010)', '(021)<-(000)'] # J=Kc doublets
v_bands = [['(101)<-(000)',0], ['(021)<-(000)',1], ['(200)<-(000)',2]] # most common vibrational transitions
ka_all = [[0,0, 'o','solid'], [0,1, 'x','dotted'], [1,0, '+','dashed'], [1,1, '1','dashdot']]

props_which = [['nu','Line Position, $\\nu$ [cm$^{-1}$]'],
               ['sw','Line Strength, S$_{296}$ [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'],
               ['gamma_self','Self-Broadened Half Width, $\gamma_{self}$ [cm$^{-1}$/atm]'],
               ['n_self','Temp. Dep. of Self-Broadened Half Width, n$_{self}$'],
               ['sd_self','Self Broadening Speed Dependence, a$_{w}$=$\Gamma_{2}$/$\Gamma_{0}$'],
               ['delta_self','Self Pressure Shift, $\delta_{self}$(T$_{0}$) [cm$^{-1}$/atm]'],
               ['n_delta_self','Temp. Dep. of Self Shift, $\delta^{\'}_{self}$ [cm$^{-1}$/(atm$\cdot$K]'],
               ['gamma_air','Air-Broadened Half Width, $\gamma_{air}$ [cm$^{-1}$/atm]'],
               ['n_air','Temp. Dep. of Air-Broadened Half Width, n$_{air}$'],
               ['sd_air','Air Broadening Speed Dependence, a_${w}$=$\Gamma_{2}$/$\Gamma_{0}$'],
               ['delta_air','Air Pressure Shift, $\delta_{air}$(T$_{0}$) [cm$^{-1}$/atm]'],
               ['n_delta_air','Temp. Dep. of Air Shift, $\delta^{\'}_{air}$ [cm$^{-1}$/(atm$\cdot$K]']]


for v_band, k in v_bands: 
    
    label1 = v_band 

    for ka in ka_all: 

        if v_bands.index([v_band, k]) == 0: label2 = 'Kc=J, Ka\'='+str(ka[0])+', Ka\'\'='+str(ka[1])
        else: label2 = ''
                
        df_sceg2 = df_sceg.copy()
        keep_boolean = ((~df_sceg2.index.isin(features_doublets)) & (df_sceg2.index.isin(features_clean)) # no doublets, only clean features
                         & (df_sceg2.index < 100000) # no nea features
                         & (df_sceg2.Jpp == df_sceg2.Kcpp) & (df_sceg2.Jp == df_sceg2.Kcp) # kc
                         & (df_sceg2.Kapp == ka[0]) & (df_sceg2.Kap == ka[1]) # ka
                         & (df_sceg2.v_all == v_band)) # vibrational band
                         
        df_sceg2 = df_sceg2[keep_boolean]
        df_og2 = df_og[keep_boolean]
        
        j=1
        
        for prop, plot_y_label in props_which:
            
            # don't bother with these guys ['elower','quanta','nu','sw']
        
            df_plot = df_sceg2[df_sceg2['uc_'+prop] > 0]
            df_plot_og = df_ht2020.loc[df_plot.index-1]
            
            # print(prop)
            # print('    ' + str(len(df_sceg[df_sceg['uc_'+prop] > 0]))) # number of total features that were fit
            
            plot_x = df_plot.m # + (df_plot.Kapp/df_plot.Jpp)/2
            plot_y = df_plot[prop]
            plot_y_unc = df_plot['uc_'+prop]
                        
            plt.figure(3*j-2, figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            # plt.plot(plot_x,plot_y,marker='x', markerfacecolor='None',linestyle='None', color=colors[0])
            plt.plot(plot_x,plot_y,marker=ka[2], markerfacecolor='None',linestyle='None', color=colors[k],  label=label1)
            # plt.plot(plot_x,plot_y,marker=ka[2], markerfacecolor='None',linestyle='None', color=colors[k],  label=label2) # plot for the legend
            # plt.plot(plot_x,plot_y,marker='None', markerfacecolor='None',linestyle=ka[3], color=colors[k],  label=label2) # plot for the legend
            plt.errorbar(plot_x,df_plot[prop], yerr=plot_y_unc, color=colors[0], ls='none')
            try: 
                plot_y_og = df_plot_og[prop]
                plt.plot(plot_x,plot_y_og,linestyle=ka[3], color=colors[k])
            except: pass
            
            plt.xlabel('m') # ' + Ka\'\'/J\'\'/2')
            plt.ylabel(plot_y_label)
            # plt.legend(edgecolor='k')       
            

            #for i in df_plot.index:
            #    i = int(i)
            #    plt.annotate(str(i),(plot_x[i], plot_y[i]))

            r'''
            plt.figure(3*j-1, figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            plt.plot(abs(plot_x),plot_y,marker=ka[2], markerfacecolor='None',linestyle='None', color=colors[k],  label=label1)
            plt.plot(abs(plot_x),plot_y_og,linestyle=ka[3], color=colors[k])
            plt.errorbar(abs(plot_x),df_plot[prop], yerr=plot_y_unc, color=colors[k], ls='none')
            plt.xlabel('|m| + Ka\'\'/J\'\'/2')
            plt.ylabel(plot_y_label)
            plt.legend(edgecolor='k')        
            r'''     
            r'''
            plt.figure(3*j, figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
            plt.plot(plot_x,plot_y-plot_y_og,marker=ka[2],linestyle='None',  color=colors[k], label=label1)
            plot_unc_y = df_plot.uc_delta_air
            plt.errorbar(plot_x,plot_y-plot_y_og, yerr=plot_y_unc,  color=colors[k], ls='none')
            plt.xlabel('m + Ka\'\'/J\'\'/2')
            plt.ylabel('('+prop+' Labfit) - ('+prop+' HITRAN 2020)')
            plt.legend(edgecolor='k')    
            r'''
            
            j+=1
        
        label1 = ''
        



# %% percent change SW with E lower

buffer = 1.3

df_plot = df_sceg[df_sceg.uc_sw > 0].sort_values(by=['elower'])
df_plot_og = df_og[df_sceg.uc_sw > 0].sort_values(by=['elower'])

label_x = 'Line Strength, S$_{296}$ (updated) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Line Strength Percent Change, S$_{296}$ \n (New - HITRAN) / HITRAN [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_y'] = (df_plot.sw - df_plot_og.sw) / df_plot_og.sw
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_sw
label_c = 'Lower State Energy [cm$^{-1}$]'
plot_c = df_plot.elower

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.errorbar(plot_x,plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2)

df_plot.sort_values(by=['sw'], inplace=True)

for fit_order in [6,7,8]:
    a = np.log10(df_plot['plot_x']*np.power(10,20))
    b = np.polyfit(a, df_plot['plot_y'], fit_order)
    c = np.poly1d(b)
    d = c(a)
    plt.plot(df_plot['plot_x'], d, 'r')

plt.hlines(0,min(plot_x), max(plot_x),linestyles='dashed')

plt.xlim(min(plot_x)/buffer, max(plot_x)*buffer)
plt.xscale('log')

plt.colorbar(sc, label=label_c)

plt.show()



# %% change in line center with E lower

buffer = 1.3

df_plot = df_sceg[df_sceg.uc_nu > 0].sort_values(by=['elower'])
df_plot_og = df_og[df_sceg.uc_nu > 0].sort_values(by=['elower'])

label_x = 'Line Strength, S$_{296}$ (updated) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'
df_plot['plot_x'] = df_plot.sw
plot_x = df_plot['plot_x']

label_y = 'Line Position Difference, $\Delta\\nu$ (New - HITRAN) [cm$^{-1}$]'
df_plot['plot_y'] = df_plot.nu - df_plot_og.nu
plot_y = df_plot['plot_y']

plot_y_unc = df_plot.uc_nu
label_c = 'Lower State Energy [cm$^{-1}$]'
plot_c = df_plot.elower

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.errorbar(plot_x,plot_y, yerr=plot_y_unc, color='k', ls='none', zorder=1)
sc = plt.scatter(plot_x, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2)

df_plot.sort_values(by=['sw'], inplace=True)

for fit_order in [6,7,8]:
    a = np.log10(df_plot['plot_x']*np.power(10,20))
    b = np.polyfit(a, df_plot['plot_y'], fit_order)
    c = np.poly1d(b)
    d = c(a)
    plt.plot(df_plot['plot_x'], d, 'r')

plt.hlines(0,min(plot_x), max(plot_x),linestyles='dashed')

plt.xlim(min(plot_x)/buffer, max(plot_x)*buffer)
plt.xscale('log')

plt.colorbar(sc, label=label_c)

plt.show()





#%% plot of air-water  spectra with residuals

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')

[T_all, P_all] = np.asarray([T_air, P_air])

T_plot = 300
P_plots = [599, 319.5, 160, 120,  80,  60,  40, 20.5]
# [160, 120,  80,  60,  40, 20.5] # [599, 319.5, 160, 120,  80,  60,  40, 20.5] # [599, 319.5, 160, 80, 40]

for P_plot in P_plots:
    
    j = P_plots.index(P_plot)
    print(j)
    i_plot = np.where((T_all == T_plot) & (P_all == P_plot))[0]
        
    T = [T_all[i] for i in i_plot]
    P = [P_all[i] for i in i_plot]
    wvn = np.concatenate([wvn_air[i] for i in i_plot])
    trans = np.concatenate([trans_air[i] for i in i_plot])
    res = np.concatenate([res_air[i] for i in i_plot])
    res_og = np.concatenate([res_og_air[i] for i in i_plot])
    
    plt.plot(wvn, trans, color=colors_grad[j], label=str(T_plot) + ', ' + str(P_plot))
    plt.plot(wvn, res+105, color=colors_grad[j])
    plt.plot(wvn, res_og+110, color=colors_grad[j])
    plt.plot(wvn, res-res_og+115, color=colors_grad[j])

plt.title('air ' + str(T_plot))





#%% plot of pure water spectra with residuals


plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')

[T_all, P_all] = np.asarray([T_pure, P_pure])

T_plot = 300
P_plots = [16, 8, 4, 3, 2, 1.5, 1, 0.5]
# 

for P_plot in P_plots:
    
    j = P_plots.index(P_plot)
    print(j)
    i_plot = np.where((T_all == T_plot) & (P_all == P_plot))[0]
        
    T = [T_all[i] for i in i_plot]
    P = [P_all[i] for i in i_plot]
    wvn = np.concatenate([wvn_pure[i] for i in i_plot])
    trans = np.concatenate([trans_pure[i] for i in i_plot])
    res = np.concatenate([res_pure[i] for i in i_plot])
    res_og = np.concatenate([res_og_pure[i] for i in i_plot])
    
    plt.plot(wvn, trans, color=colors_grad[j], label=str(T_plot) + ', ' + str(P_plot))
    plt.plot(wvn, res+105, color=colors_grad[j])
    plt.plot(wvn, res_og+110, color=colors_grad[j])
    plt.plot(wvn, res-res_og+115, color=colors_grad[j])

plt.title('air ' + str(T_plot))



# %% new features and features below NF

plot_which_y = 'sw'
plot_which_y_extra = ''
label_y = 'Line Strength, S$_{296}$ (updated) [cm$^{-1}$/(molecule$\cdot$cm$^{-2}$)]'

plot_which_x = 'elower'
label_x = 'Lower State Energy [cm$^{-1}$]'

df_plot = df_sceg[df_sceg.index > 10000]

plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y + plot_which_y_extra]

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.plot(plot_x, plot_y, 'kx', label = 'other features')

if plot_clean: 
    
    df_plot_clean = df_plot[df_plot.index.isin(features_clean)]
    plot_x_clean = df_plot_clean[plot_which_x]
    plot_y_clean = df_plot_clean[plot_which_y + plot_which_y_extra]
    
    plt.plot(plot_x_clean, plot_y_clean, 'rx', label = 'isolated features')
    plt.legend()
    
if plot_doublets: 
    
    df_plot_doublets = df_plot[df_plot.index.isin(features_doublets)]
    plot_x_doublet = df_plot_doublets[plot_which_x]
    plot_y_doublet = df_plot_doublets[plot_which_y + plot_which_y_extra]
    
    plt.plot(plot_x_doublet, plot_y_doublet, 'g+', label = 'doublets')
    plt.legend()

if plot_unc_x: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color='k', ls='none')
if plot_unc_y: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color='k', ls='none')

if plot_labels:
    for j in df_plot.index:
        j = int(j)
        plt.annotate(str(j),(plot_x[j], plot_y[j]))

if plot_logx: 
    plt.xscale('log')





















# %% temp dependence of shift (settings from previous plots)



plot_which_y = 'n_delta_self'
plot_which_y_extra = ''
label_y = 'Temperature Dependence of Pressure Shift'
plot_unc_y = True

plot_clean = True
plot_doublets = True
plot_labels = False
plot_logx = False

plot_which_x = 'v1pp'
label_x = '(Jp+Jpp) / 2'
plot_unc_x = False

df_plot = df_compare_all[df_compare_all['uc_'+plot_which_y] > -1]

plot_x = df_plot[plot_which_x]
plot_y = df_plot[plot_which_y + plot_which_y_extra]

plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
plt.xlabel(label_x)
plt.ylabel(label_y)

plt.plot(plot_x, plot_y, 'kx', label = 'other features')

if plot_clean: 
    
    df_plot_clean = df_plot[df_plot.index.isin(features_clean)]
    plot_x_clean = df_plot_clean[plot_which_x]
    plot_y_clean = df_plot_clean[plot_which_y + plot_which_y_extra]
    
    plt.plot(plot_x_clean, plot_y_clean, 'rx', label = 'isolated features')
    plt.legend()
    
if plot_doublets: 
    
    df_plot_doublets = df_plot[df_plot.index.isin(features_doublets)]
    plot_x_doublet = df_plot_doublets[plot_which_x]
    plot_y_doublet = df_plot_doublets[plot_which_y + plot_which_y_extra]
    
    plt.plot(plot_x_doublet, plot_y_doublet, 'g+', label = 'doublets')
    plt.legend()

if plot_unc_x: 
    plot_unc_x = df_plot['uc_'+plot_which_x]
    plt.errorbar(plot_x, plot_y, xerr=plot_unc_x, color='k', ls='none')
if plot_unc_y: 
    plot_unc_y = df_plot['uc_'+plot_which_y]
    plt.errorbar(plot_x, plot_y, yerr=plot_unc_y, color='k', ls='none')

if plot_labels:
    for j in df_plot.index:
        j = int(j)
        plt.annotate(str(j),(plot_x[j], plot_y[j]))

if plot_logx: 
    plt.xscale('log')







