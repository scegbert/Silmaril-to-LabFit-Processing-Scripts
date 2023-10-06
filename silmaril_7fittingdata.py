
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score




def sawtooth_wave(x, a2, y_max, slope, offset):
    
    x1, x2 = x   
    
    y = slope * ((x1 + a2*x2) % 1) + offset
    
    # y[y > y_max] = y_max
    return y
    
def func(x, j1, j2, k1, k2, c):
    
    x1, x2 = x   
    
    j2 = 0
    
    return c + j1*x1 + j2*x1**2 + k1*x2 + k2*x2**2

def func3(x, j1, j2, k1, k2, k3, e1, e2, c):
    
    x1, x2, x3 = x   
        
    return j1 * np.exp(-j2*x1) + 0.04 + k1*x3 + k2*x3**2 + k3*x3**3 + e1*x2 + e2*x2**2

def pade(x, a1, a2, a3, a4):
    
    x1, x2, x3 = x   
        
    return a1 / (a2*x1-a3) + a4

def expon(x, a1, a2, a3, a4):
    
    x1, x2, x3 = x   
        
    return a1 * np.exp(-a2*x1) + a3
    



#%% gamma
# breakdown data

# plot_which_y = 'gamma_self'
# which = (df_sceg['uc_gamma_self'] > -1) # &(df_sceg['uc_n_self'] > -1) # &(df_sceg['Kapp'] < 5)

plot_which_y = 'gamma_air'
which = (df_sceg['uc_gamma_air'] > -1) # &(df_sceg['uc_n_air'] > -1) # &(df_sceg['Kapp'] < 5)


df_plot = df_sceg_align[which] # floating all width parameters

plot_x3 = df_plot['Kcpp'].to_numpy()

plot_x1 = df_plot['Jpp'].to_numpy()

plot_x2 = df_plot['elower'].to_numpy()

# digit_lists = [list(map(int, string)) for string in df_plot['vp'].to_numpy()]
# digit_arrays = np.array(digit_lists)
# result_array = np.sum(digit_arrays, axis=1) - digit_arrays[:,1] / 2
# plot_x2 = result_array.copy()

plot_x_fit = [plot_x1, plot_x2, plot_x3]


plot_y = df_plot[plot_which_y].to_numpy()
plot_c = df_plot['Jpp']


a1 = 0.22
a2 = 0.033
a3 = 0.01
a4 = .2

# Fit the data to the sawtooth_wave function
initial_guess = [a1, a2, a3, a4]  # Initial guess for frequency, phase shift, and slope
fit_params, _ = curve_fit(expon, plot_x_fit, plot_y, p0=initial_guess)#, bounds=bounds)

a1, a2, a3, a4 = fit_params

fit_y = expon(plot_x_fit, a1, a2, a3, a4)

plt.figure()
sc = plt.scatter(plot_x1, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
plt.plot(plot_x1, fit_y, 'kx')


mad = np.mean(np.abs(plot_y-fit_y))
rms = np.sqrt(np.sum((plot_y-fit_y)**2)/ len(plot_y))
r2 = r2_score(plot_y, fit_y)

print('{}    {}     {}\n\n\n'.format(mad, rms, r2))


#%% temp dep of width n_gamma

# breakdown data

# plot_which_y = 'gamma_self'
# which = (df_sceg['uc_gamma_self'] > -1) # &(df_sceg['uc_n_self'] > -1) # &(df_sceg['Kapp'] < 5)

plot_which_y = 'n_air'
which = (df_sceg['uc_n_air'] > -1) # &(df_sceg['uc_n_air'] > -1) # &(df_sceg['Kapp'] < 5)


df_plot = df_sceg_align[which] # floating all width parameters

plot_x3 = df_plot['Kcpp'].to_numpy()

plot_x1 = df_plot['Jpp'].to_numpy()

plot_x2 = df_plot['elower'].to_numpy()

# digit_lists = [list(map(int, string)) for string in df_plot['vp'].to_numpy()]
# digit_arrays = np.array(digit_lists)
# result_array = np.sum(digit_arrays, axis=1) - digit_arrays[:,1] / 2
# plot_x2 = result_array.copy()

plot_x_fit = [plot_x1, plot_x2, plot_x3]


plot_y = df_plot[plot_which_y].to_numpy()
plot_c = df_plot['Jpp']


a1 = -0.01
a2 = -0.3
a3 = 1
a4 = .2

# Fit the data to the sawtooth_wave function
initial_guess = [a1, a2, a3, a4]  # Initial guess for frequency, phase shift, and slope
fit_params, _ = curve_fit(expon, plot_x_fit, plot_y, p0=initial_guess)#, bounds=bounds)

a1, a2, a3, a4 = fit_params

fit_y = expon(plot_x_fit, a1, a2, a3, a4)

plt.figure()
sc = plt.scatter(plot_x1, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)
plt.plot(plot_x1, fit_y, 'kx')


mad = np.mean(np.abs(plot_y-fit_y))
rms = np.sqrt(np.sum((plot_y-fit_y)**2)/ len(plot_y))
r2 = r2_score(plot_y, fit_y)

print('{}    {}     {}\n\n\n'.format(mad, rms, r2))





















