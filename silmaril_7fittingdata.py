
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



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
        
    return c + j1*x1 + j2*x1**2 + k1*x3 + k2*x3**2 + k3*x3**3 + e1*x2 + e2*x2**2
    

#%%
# breakdown data

plot_which_y = 'gamma_self'

which = (df_sceg['uc_gamma_self'] > -1) # &(df_sceg['uc_n_self'] > -1) # &(df_sceg['Kapp'] < 5)
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


# #%%

# j1 = 1
# j2 = 1
# k1 = 0
# k2 = 0
# c = 0

# y = func(plot_x_fit, j1, j2, k1, k2, c)

# #%%

j1 = -3.43293715e-02
j2 = 8.32443988e-04
k1 = 3.94669192e-02
k2 = -3.97332291e-03
k3 = 1.05067590e-04
e1 = -6.21037034e-05
e2 = 1.55277520e-08
c = 5.03437782e-01


# Fit the data to the sawtooth_wave function
initial_guess = [j1, j2, k1, k2, k3, e1, e2, c]  # Initial guess for frequency, phase shift, and slope
fit_params, _ = curve_fit(func3, plot_x_fit, plot_y, p0=initial_guess)#, bounds=bounds)

j1, j2, k1, k2, k3, e1, e2, c = fit_params

fit_y = func3(plot_x_fit, j1, j2, k1, k2, k3, e1, e2, c)

plt.figure()
plt.plot(plot_x1, fit_y, 'x')

sc = plt.scatter(plot_x1, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)

mad = np.mean(np.abs(plot_y-fit_y))
rms = np.sqrt(np.sum((plot_y-fit_y)**2)/ len(plot_y))
r2 = r2_score(plot_y, fit_y)

print('{}    {}     {}'.format(mad, rms, r2))

#%%

a2 = 0.052
y_max = 1
slope = -1.8
offset = 1


# Fit the data to the sawtooth_wave function
initial_guess = [a2, y_max, slope, offset]  # Initial guess for frequency, phase shift, and slope
bounds = [(a2*0.9, y_max*0.9, slope*1.1, offset*0.9), (a2*1.1, y_max*1.1, slope*0.9, offset*1.1)]
fit_params, _ = curve_fit(sawtooth_wave, plot_x_fit, plot_y, p0=initial_guess)#, bounds=bounds)

a2, y_max, slope, offset = fit_params

plot_1x = plot_x1 + plot_x2*a2
plot_1x = np.linspace(0, 10, num=1000)
y = slope * ((plot_1x) % 1) + offset
y[y > y_max] = y_max


plt.figure()
plt.plot(plot_1x, y)

plot_x_og = plot_x1 + a2*plot_x2
sc = plt.scatter(plot_x_og, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)


#%%

a2 = 0.052
y_max = 0.44
slope = -0.67
offset = 0.63

# a2 = 0.043
# y_max = 10
# slope = -1.54
# offset = 1.0


plt.figure()

plot_1x = plot_x1*a1 + plot_x2*a2
plot_1x.sort()
y_1x = (plot_1x + shift) % a3 * slope + offset


plot_1x = np.linspace(0, 15, num=1000)
y_smooth = (plot_1x + shift) % 1 * slope + offset
y_smooth[y_smooth > y_max] = y_max
y_smooth[y_smooth < 0.1] = 0.1


mad = np.mean(np.abs(plot_y-y_1x))
rms = np.sqrt(np.sum((plot_y-y_1x)**2)/ len(plot_y))
r2 = r2_score(plot_y, y_1x)


plt.plot(plot_1x, y_smooth, label = str(np.round(mad, 3))[:5])

plot_x_og = a1*plot_x1 + a2*plot_x2
sc = plt.scatter(plot_x_og, plot_y, marker='x', c=plot_c, cmap='viridis', zorder=2, linewidth=2, vmin=0, vmax=16)


plt.legend()






















