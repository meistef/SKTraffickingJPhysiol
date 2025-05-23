#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Prof. Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 03/11/2023
"""
#%% Import packages and set directories.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import myokit
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
from scipy.stats import iqr
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import os
import time
import ast

# Set directories. Note, the directories need to be set correctly on your own device.
work_dir = os.getcwd() 
work_dir = os.path.join(work_dir, 'Documents', 'GitHub', 'SK_Model')
os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)
    
# Import local functions.    
from SKTraffickingModelFunctions import add_beat, time_freq, triangular_pacing, trian_square_pacing, figure_1C, dem_reg_pacing, plot_dem_reg, max_ftrap, plot_koca, incub_prot, current_prepace, find_peaks_in_data, ftrap_freq_export, calc_apd90, voltage_clamp_prot, reversibility, export_PoM_results_by_repeat, PoM, correct_time_glitch, PoM_sens_PLS, visualize_PoM_outputs, drug_effects, current_prepace_drugs, create_block_df_apd, create_block_df
#%% Calcium dependence

# Given that SK functioning and trafficking are Ca2+ dependent, we want to evaluate the effects of the different
# Ca2+ components at different pacing frequencies (1 Hz, 3 Hz, 5Hz).

# Load the model in global space. 
m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

# Define a list of frequencies. 
freq = [0.1, 1, 2, 3, 4, 5]

# Define the Ca2+ buffer
ca_i = 0.0005

# Run the incubation protocol.
pp = 10000000
incub_dem = incub_prot(pp = pp, ca_i = ca_i)
incub_free = incub_prot(pp = pp, ca_i = None)

# Run the triangular pacing for 10 minutes. 
trian_reg = triangular_pacing(t = 10, freq = freq, ca_i = None, incub = incub_free['incub_v'])    
trian_dem = triangular_pacing(t = 10, freq = freq, ca_i = ca_i, incub = incub_dem['incub_vcai'])   

# Plot the Ca2+ concentration (ftrap)
fig, ax = plt.subplots(2, 1)  
for i, result in enumerate(trian_reg['max_ftrap']):
    ax[0].plot(result['Time'], result['Peak Value'], label=f'Frequency: {freq[i]} Hz')
ax[0].set_xlabel('Time (min)')
ax[0].set_ylabel('Ca Concentration (fTrap)')
ax[0].set_title('fTrap for different frequencies (regular)')
#ax[0].legend(loc = 'center right')

for i, result in enumerate(trian_dem['max_ftrap']):
    ax[1].plot(result['Time'], result['Peak Value'], label=f'Frequency: {freq[i]} Hz')
ax[1].set_xlabel('Time (min)')
ax[1].set_ylabel('Ca Concentration (fTrap)')
ax[1].set_title(f'fTrap for different frequencies (Ca2+ buffer = {ca_i} mM)')
#ax[1].legend(loc = 'center right')

# Create a shared legend below the plots
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

plt.tight_layout()
plt.show()

# Obtain the maximum of each fTrap.
max_ftrap_reg = [max(result['Ca_Concentrations.fTrap']) for result in trian_reg['d_list']]
max_ftrap_dem = [max(result['Ca_Concentrations.fTrap']) for result in trian_dem['d_list']]

# Plot the maximum fTraps.
plt.figure()
plt.plot(freq, max_ftrap_reg, label='Regular Ca2+')
plt.plot(freq, max_ftrap_dem, label=f'Ca2+ buffer = {ca_i} mM')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Max Ca2+ Concentration (fTrap)')
plt.legend(loc='upper left')
plt.title('Frequency-dependence of fTrap')
plt.show()

# Combine the lists into one DataFrame
max_ftrap_combined = pd.DataFrame({
    'Frequency (Hz)': freq,
    'Max Regular Ca2+ (fTrap)': max_ftrap_reg,
    'Max Ca2+ Buffer (fTrap)': max_ftrap_dem})
  
# Export to Prism
ftrap_freq_export(dictionary = trian_reg, data = 'max_ftrap', freq = freq, exp_name = 'freq_dep_regV2')
ftrap_freq_export(dictionary = trian_dem, data = 'max_ftrap', freq = freq, exp_name = 'freq_dep_demV2')
max_ftrap_combined.to_csv('Results/max_ftrap_freqV2.csv', index = False)
#%% Re-create the plots from Figure 7 of Heijman et al. (2023). Circ Res.

# Run the simulation for 10 minutes of triangular pacing and then a square voltage clamp protocol at 0.1 and 5 Hz pacing.
# Also run the baseline square pulse voltage clamp procotol.
sim_01 = trian_square_pacing(t = 10, freq = 0.1, ca_i = ca_i, incub_state = incub_dem['incub_vcai'])
sim_5 = trian_square_pacing(t = 10, freq = 5, ca_i = ca_i, incub_state = incub_dem['incub_vcai'])
sim_base = figure_1C(ca_i = ca_i, incub_state = incub_dem['incub_vcai'])

# Set the +80 mV results in a dataframe.
max_80 = pd.DataFrame({'Frequency': ['baseline','0.1 Hz', '5 Hz'],'Max SK Value': [sim_base['max_sk'][-1], sim_01['max_sk'][-1], sim_5['max_sk'][-1]]})

# Plot the results
plt.figure()
plt.plot(np.array(sim_01['trian']['engine.time'])/60000, sim_01['trian']['I_SK_trafficking.S'], label = 'Sub (0.1 Hz)')
plt.plot(np.array(sim_01['trian']['engine.time'])/60000, sim_01['trian']['I_SK_trafficking.M'], label = 'Mem (0.1 Hz)')
plt.plot(np.array(sim_5['trian']['engine.time'])/60000, sim_5['trian']['I_SK_trafficking.S'], label = 'Sub (5 Hz)')
plt.plot(np.array(sim_5['trian']['engine.time'])/60000, sim_5['trian']['I_SK_trafficking.M'], label = 'Mem (5 Hz)')
plt.legend(loc = 'center left')
plt.xlabel('Time [min]')
plt.ylabel('Number of channels')
plt.tight_layout()

# Plot the ion channel trafficking.
fig, ax = plt.subplots(3, 1)
ax[0].plot(np.array(sim_01['trian']['engine.time'])/60000, sim_01['trian']['I_SK.I_sk'], label = '0.1 Hz')
ax[0].set_ylabel('pA/pF')
ax[0].set_title(f'0.1 Hz (Triangular pacing; Ca2+ buffer = {ca_i} mM)')
ax[0].set_xlabel('Time [min]')

ax[1].plot(np.array(sim_5['trian']['engine.time'])/60000, sim_5['trian']['I_SK.I_sk'], label = '5 Hz')
ax[1].set_ylabel('pA/pF')
ax[1].set_xlabel('Time [min]')
ax[1].set_title(f'5 Hz (Triangular pacing; Ca2+ buffer = {ca_i} mM)')

sns.barplot(x = 'Frequency', y = 'Max SK Value', hue = 'Frequency', data = max_80, ax = ax [2], palette = ['blue', 'orange', 'red'])
ax[2].set_title('Voltage clamp protocol after pacing (at +80 mV)')
ax[2].set_xlabel('')
ax[2].set_ylabel('ISK')

plt.tight_layout()
plt.show()

# Repeat these simulations for free Ca2+
sim_01_free = trian_square_pacing(t = 10, freq = 0.1, ca_i = None, incub_state = incub_free['incub_v'])
sim_5_free = trian_square_pacing(t = 10, freq = 5, ca_i = None, incub_state = incub_free['incub_v'])
sim_base_free = figure_1C(ca_i = None, incub_state = incub_free['incub_v'])

# Set the +80 mV results in a dataframe.
max_80_free = pd.DataFrame({'Frequency': ['baseline','0.1 Hz', '5 Hz'],'Max SK Value': [sim_base_free['max_sk'][-1], sim_01_free['max_sk'][-1], sim_5_free['max_sk'][-1]]})

# Plot the results
plt.figure()
plt.plot(np.array(sim_01_free['trian']['engine.time'])/60000, sim_01_free['trian']['I_SK_trafficking.S'], label = 'Sub (0.1 Hz)')
plt.plot(np.array(sim_01_free['trian']['engine.time'])/60000, sim_01_free['trian']['I_SK_trafficking.M'], label = 'Mem (0.1 Hz)')
plt.plot(np.array(sim_5_free['trian']['engine.time'])/60000, sim_5_free['trian']['I_SK_trafficking.S'], label = 'Sub (5 Hz)')
plt.plot(np.array(sim_5_free['trian']['engine.time'])/60000, sim_5_free['trian']['I_SK_trafficking.M'], label = 'Mem (5 Hz)')
plt.legend(loc = 'center left')
plt.xlabel('Time [min]')
plt.ylabel('Number of channels')
plt.title('Free Ca2+')
plt.tight_layout()

# Plot the ion channel trafficking.
fig, ax = plt.subplots(3, 1)
ax[0].plot(np.array(sim_01_free['trian']['engine.time'])/60000, sim_01_free['trian']['I_SK.I_sk'], label = '0.1 Hz')
ax[0].set_ylabel('pA/pF')
ax[0].set_title(f'0.1 Hz (Triangular pacing; free Ca2+)')
ax[0].set_xlabel('Time [min]')

ax[1].plot(np.array(sim_5_free['trian']['engine.time'])/60000, sim_5_free['trian']['I_SK.I_sk'], label = '5 Hz')
ax[1].set_ylabel('pA/pF')
ax[1].set_xlabel('Time [min]')
ax[1].set_title(f'5 Hz (Triangular pacing; free Ca2+)')

sns.barplot(x = 'Frequency', y = 'Max SK Value', hue = 'Frequency', data = max_80_free, ax = ax [2], palette = ['blue', 'orange', 'red'])
ax[2].set_title('Voltage clamp protocol after pacing (at +80 mV)')
ax[2].set_xlabel('')
ax[2].set_ylabel('ISK')

plt.tight_layout()
plt.show()

# Combine the data into a dataframe.
dem_01hz = pd.DataFrame({'Time': np.array(sim_01['trian']['engine.time'])/60000,
                         'ISK': sim_01['trian']['I_SK.I_sk']}).iloc[::20, :]
dem_5hz = pd.DataFrame({'Time': np.array(sim_5['trian']['engine.time'])/60000,
                         'ISK': sim_5['trian']['I_SK.I_sk']}).iloc[::20, :]
free_01hz = pd.DataFrame({'Time': np.array(sim_01_free['trian']['engine.time'])/60000,
                         'ISK': sim_01_free['trian']['I_SK.I_sk']}).iloc[::20, :]
free_5hz = pd.DataFrame({'Time': np.array(sim_5_free['trian']['engine.time'])/60000,
                         'ISK': sim_5_free['trian']['I_SK.I_sk']}).iloc[::20, :]

# Export data to Prism.
dem_01hz.to_csv('Results/dem_01hz.csv', index = False)
dem_5hz.to_csv('Results/dem_5hz.csv', index = False)
free_01hz.to_csv('Results/free_01hz.csv', index = False)
free_5hz.to_csv('Results/free_5hz.csv', index = False)
max_80.to_csv('Results/Figure7_max80_dem.csv', index = False)
max_80_free.to_csv('Results/Figure7_max80_free.csv', index = False)

#%% Re-create the plots from Figure 1C of Heijman et al. (2023). Circ Res.

# Run the protocols for Figure 1C and 5 Hz triangular pacing.
figure1C = figure_1C(ca_i = ca_i, incub_state = incub_dem['incub_vcai'])
free_IV = figure_1C(ca_i = None, incub_state = incub_free['incub_v'])
demote_5hz = trian_square_pacing(t = 10, freq = 5, ca_i = ca_i, incub_state = incub_dem['incub_vcai'])
freeca_5hz = trian_square_pacing(t = 10, freq = 5, ca_i = None, incub_state = incub_free['incub_v'])

# The IV curves and Action Potentials.
volt_steps = np.arange(-120, 90, 10)

# Load the experimental data.
exp1C= pd.read_csv('Figures/Figure1C_sum.csv')

# Change the first 4 column names
exp1C_colnames = ['voltage', 'mean', 'std', 'n'] + list(exp1C.columns[4:])
exp1C.columns = exp1C_colnames

# Plot the peak ISK without pacing and after pacing.
plt.figure()
plt.plot(volt_steps, figure1C['max_sk'], label='No pacing (500 nM Cai)')
plt.errorbar(volt_steps, exp1C['mean'], yerr=exp1C['std']/2, fmt='o', label='No pacing (exp. 500 nM Cai)', capsize=5)
plt.plot(volt_steps, demote_5hz['max_sk'], label='After 5 Hz pacing (10 min; 500 nM Cai)')
plt.xlabel('Voltage steps (mV)')
plt.ylabel('Current density (pA/pF)')
plt.title('ISK')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the IV for ISK with free Ca2+
plt.figure()
plt.plot(volt_steps, free_IV['max_sk'], label = 'No pacing (Free Ca2+)')
plt.plot(volt_steps, freeca_5hz['max_sk'], label='After 5 Hz pacing (10 min; Free Ca2+)')
plt.xlabel('Voltage steps (mV)')
plt.ylabel('Current density (pA/pF)')
plt.title('ISK')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the IV for demote and free Ca2+ post pacing.
plt.figure()
plt.plot(volt_steps, freeca_5hz['max_sk'], label = 'free')
plt.plot(volt_steps, demote_5hz['max_sk'], label = 'demote')
plt.legend()

plt.figure()
plt.plot(freeca_5hz['trian']['engine.time'], freeca_5hz['trian']['I_SK.I_sk'])
plt.plot(demote_5hz['trian']['engine.time'], demote_5hz['trian']['I_SK.I_sk'])

# Repeat the 5 Hz pacing under demote (and free) + LTCC block.
demote_5hz_b = trian_square_pacing(t = 10, freq = 5, ca_i = ca_i, incub_state = incub_dem['incub_vcai'], ca_block = True)
free_5hz_b = trian_square_pacing(t = 10, freq = 5, ca_i = None, incub_state = incub_free['incub_v'], ca_block = True)

# Combine the dataframes.
demote_exp_80 = pd.DataFrame({
    'Mean': [figure1C['max_sk'][-1], demote_5hz['max_sk'][-1], demote_5hz_b['max_sk'][-1], 0.981, 7.273, 3.094],
    'Std': [0, 0, 0, 0.898, 2.642, 1.026],
    'Cond': ['Baseline', '5 Hz', '5 Hz w/ block', 'Baseline', '5 Hz', '5 Hz w/ block'],
    'Mode': ['Model', 'Model', 'Model', 'Exp', 'Exp', 'Exp']})

free_exp_80 = pd.DataFrame({
    'Mean': [sim_base_free['max_sk'][-1], sim_5_free['max_sk'][-1], free_5hz_b['max_sk'][-1], 0.981, 7.273, 3.094],
    'Std': [0, 0, 0, 0.898, 2.642, 1.026],
    'Cond': ['Baseline', '5 Hz', '5 Hz w/ block', 'Baseline', '5 Hz', '5 Hz w/ block'],
    'Mode': ['Model', 'Model', 'Model', 'Exp', 'Exp', 'Exp']})


# Plot the demote/buffer results.
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(np.array(demote_5hz['trian']['engine.time']) / 60000, demote_5hz['trian']['I_SK.I_sk'], label='demote')
ax[0].set_ylabel('pA/pF')
ax[0].set_title('5 Hz (Triangular pacing)')
ax[0].set_xlabel('Time [min]')

ax[1].plot(np.array(demote_5hz_b['trian']['engine.time']) / 60000, demote_5hz_b['trian']['I_SK.I_sk'], label='demote block')
ax[1].set_ylabel('pA/pF')
ax[1].set_xlabel('Time [min]')
ax[1].set_title('5 Hz (Triangular pacing) w/ LTCC block')

# Plot the barplot
sns.barplot(x='Cond', y='Mean', hue='Mode', data=demote_exp_80, ax=ax[2], palette='CMRmap_r')

# Add error bars only for the experimental data
exp_data = demote_exp_80[demote_exp_80['Mode'] == 'Exp']
exp_indices = exp_data.index.tolist()

# Calculate x_coords for experimental data
x_coords = []
for i, p in enumerate(ax[2].patches):
    if i in exp_indices:
        x_coords.append(p.get_x() + p.get_width() / 2)

ax[2].errorbar(x=x_coords, y=exp_data['Mean'], yerr=exp_data['Std'], fmt='none', c='k', capsize=5)

ax[2].set_title('Voltage clamp protocol after pacing (at +80 mV)')
ax[2].set_xlabel('')
ax[2].set_ylabel('ISK')

plt.tight_layout()
plt.show()

# Plot the free Ca2+ results
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(np.array(sim_5_free['trian']['engine.time']) / 60000, sim_5_free['trian']['I_SK.I_sk'], label='demote')
ax[0].set_ylabel('pA/pF')
ax[0].set_title('5 Hz (Triangular pacing)')
ax[0].set_xlabel('Time [min]')

ax[1].plot(np.array(free_5hz_b['trian']['engine.time']) / 60000, free_5hz_b['trian']['I_SK.I_sk'], label='demote block')
ax[1].set_ylabel('pA/pF')
ax[1].set_xlabel('Time [min]')
ax[1].set_title('5 Hz (Triangular pacing) w/ LTCC block')

# Plot the barplot
sns.barplot(x='Cond', y='Mean', hue='Mode', data=free_exp_80, ax=ax[2], palette='CMRmap_r')

# Add error bars only for the experimental data
exp_data_free = free_exp_80[free_exp_80['Mode'] == 'Exp']
exp_indices_free = exp_data_free.index.tolist()

# Calculate x_coords for experimental data
x_coords = []
for i, p in enumerate(ax[2].patches):
    if i in exp_indices_free:
        x_coords.append(p.get_x() + p.get_width() / 2)

ax[2].errorbar(x=x_coords, y=exp_data_free['Mean'], yerr=exp_data_free['Std'], fmt='none', c='k', capsize=5)

ax[2].set_title('Voltage clamp protocol after pacing (at +80 mV)')
ax[2].set_xlabel('')
ax[2].set_ylabel('ISK')

plt.tight_layout()
plt.show()

# Plot the trafficking results over time.
plt.figure()
plt.plot(np.array(demote_5hz['trian']['engine.time'])/60000, demote_5hz['trian']['I_SK_trafficking.M'], 'k', label = 'Mem.')
plt.plot(np.array(demote_5hz['trian']['engine.time'])/60000, demote_5hz['trian']['I_SK_trafficking.S'], 'red', label = 'Sub.')
plt.plot(np.array(demote_5hz_b['trian']['engine.time'])/60000, demote_5hz_b['trian']['I_SK_trafficking.M'], 'k', label = 'Mem. w/ block', ls = 'dashed')
plt.plot(np.array(demote_5hz_b['trian']['engine.time'])/60000, demote_5hz_b['trian']['I_SK_trafficking.S'], 'red', label = 'Sub. w/ block', ls = 'dashed')
plt.xlabel('Time [min]')
plt.ylabel('Number of channels')
plt.title('5 Hz (Cai = 500 nM)')
plt.legend(loc = 'upper right')
plt.tight_layout()

# Export the results to Prism.
dem_5hz_block = pd.DataFrame({'Time': np.array(demote_5hz_b['trian']['engine.time'])/60000,
                         'ISK': demote_5hz_b['trian']['I_SK.I_sk']}).iloc[::20, :]

free_5hz_block = pd.DataFrame({'Time': np.array(free_5hz_b['trian']['engine.time'])/60000,
                         'ISK': free_5hz_b['trian']['I_SK.I_sk']}).iloc[::20, :]

dem_IV = pd.DataFrame({'Voltage': volt_steps, 'No Pacing': figure1C['max_sk'],
                      'Experimental': exp1C['mean'], 'Experimental SD': exp1C['std'],
                      '5 Hz': demote_5hz['max_sk']})

free_IV = pd.DataFrame({'Voltage': volt_steps, 'No Pacing': free_IV['max_sk'],
                      '5 Hz': freeca_5hz['max_sk']})


dem_5hz_block.to_csv('Results/dem_5hz_block.csv', index = False)
free_5hz_block.to_csv('Results/free_5hz_block.csv', index = False)
demote_exp_80.to_csv('Results/demote_exp_80.csv', index = False)
free_exp_80.to_csv('Results/free_exp_80.csv', index = False)
dem_IV.to_csv('Results/dem_IV.csv', index = False)
free_IV.to_csv('Results/free_IV.csv', index = False)

#%% Perform a simulation in which the pacing frequency will change after 10 min.

def two_freq_pacing(t, freq, ca_i = None, incub = None, ca_block = False):
    
    """
    Triangular pacing function.

    Parameters:
    ----------
    t : float
        Time in minutes.
        
    freq : list
        List of frequencies (in Hz).
    
    x : int
        Optimization parameter.
   
    Returns:
    ----------
    d_list : list
        A list with myokit datalogs.
    """
    
    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

    # Get the membrane component from the model.
    c = m.get('membrane')
    
    # Get the voltage variable, and remove its binding to 'pace'.
    v = c.get('V')
    v.set_binding(None)
    v.demote()
    
    # Repeat the same for the level variable.
    l = c.get('level')
    l.set_binding(None)
    
    if ca_i is not None:
        # Set the intracellular Ca2+ concentration to 500 nM.
        cai = m.get('Ca_Concentrations.Ca_i')
        # Demote cai.
        cai.demote()
        # Set rhs to 500 nM but units are mM.
        cai.set_rhs(ca_i)

    # Define a basic cycle length of 1000 ms.
    bcl = 1000/freq
    
    # Add a v1 variable which defines the downward slope. 
    v1 = c.add_variable('v1')
    v1.set_rhs('40 - 1.2 * (engine.time % '+ str(bcl) +')')
    
    # Add a 'p' variable.
    vp = c.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')
    
    # Set a new right-hand side equation for V.
    v.set_rhs('piecewise((engine.time % ' + str(bcl) + ' >= 0 and engine.time % ' + str(bcl) + ' <= 100), v1, vp)')
    
    # Initialize a myokit protocol.
    p = myokit.Protocol()
    
    # Determine the amount of beats for the duration of simulation.
    num_beats = int(t * freq * 60)
    
    # Loop through the amount of beats for each frequency. 
    for i in range(num_beats):
        # Add the holding potential steps to the protocol.
        p.add_step(-80, 100)
        p.add_step(-80, bcl - 100)
        
    # Define the total time characteristic.
    t_trian = p.characteristic_time() - 1
        
    # Compile the simulation.
    s = myokit.Simulation(m, p)
    
    # Set the maximum timestep size and tolerances
    s.set_max_step_size(0.1)
    s.set_tolerance(1e-8, 1e-8)

    # Set the LTCC block if true.
    if ca_block is True:
        s.set_constant('I_Ca.ca_block', 0.3)
        
    # To limit automaticity under demote.
    if ca_i is not None:
        s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
 
    # Set the incubation state.
    s.set_state(incub)

    # Run the simulation for the total time of the protocol.
    d = s.run(t_trian, log=['engine.time', 'membrane.V', 'Ca_Concentrations.fTrap', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S'])
    
    # Obtain the state after the triangular pacing
    state = s.state()
    
    return dict(d = d, state = state)

# Run the protocol to evaluate the two pacing conditions for both demote and free Ca2+.
low_freq_10 = two_freq_pacing(freq = 1, t = 10, ca_i = ca_i, incub = incub_dem['incub_vcai'], ca_block = False)
high_freq_20 = two_freq_pacing(freq = 5, t = 10, ca_i = ca_i, incub = low_freq_10['state'], ca_block = False)
low_freq_10_free = two_freq_pacing(freq = 1, t = 10, ca_i = None, incub = incub_free['incub_v'], ca_block = False)
high_freq_20_free = two_freq_pacing(freq = 5, t = 10, ca_i = None, incub = low_freq_10_free['state'], ca_block = False)

# Reformat the simulation results
low_freq_time = np.array(low_freq_10['d']['engine.time'])/60000
high_freq_time = np.array(high_freq_20['d']['engine.time'])/60000
low_freq_time_free = np.array(low_freq_10_free['d']['engine.time'])/60000
high_freq_time_free = np.array(high_freq_20_free['d']['engine.time'])/60000

# Adjust high_freq_time to start from the 10-minute mark
adjusted_high_freq_time = high_freq_time + 10
adjusted_high_freq_time_free = high_freq_time_free + 10

# Plot the results. 
fig, ax = plt.subplots(1, 2)
ax[0].plot(low_freq_time, low_freq_10['d']['I_SK_trafficking.M'], 'k', label = 'Mem. (1 Hz)')
ax[0].plot(adjusted_high_freq_time, high_freq_20['d']['I_SK_trafficking.M'], 'red', label = 'Mem. (5 Hz)')
ax[0].plot(low_freq_time, low_freq_10['d']['I_SK_trafficking.S'], 'k', label = 'Sub. (1 Hz)', ls = 'dashed')
ax[0].plot(adjusted_high_freq_time, high_freq_20['d']['I_SK_trafficking.S'], 'red', label = 'Sub. (5 Hz)', ls = 'dashed')
ax[0].axvline(x = 10, color = 'grey', ls = 'dotted', label = 'Protocol change')
ax[0].set_xlabel('Time [min]')
ax[0].set_ylabel('Number of channels')
ax[0].legend(loc = 'upper left')
ax[0].set_title(f'Ca2+ buffered ({ca_i} mM)')

ax[1].plot(low_freq_time_free, low_freq_10_free['d']['I_SK_trafficking.M'], 'k', label = 'Mem. (1 Hz)')
ax[1].plot(adjusted_high_freq_time_free, high_freq_20_free['d']['I_SK_trafficking.M'], 'red', label = 'Mem. (5 Hz)')
ax[1].plot(low_freq_time_free, low_freq_10_free['d']['I_SK_trafficking.S'], 'k', label = 'Sub. (1 Hz)', ls = 'dashed')
ax[1].plot(adjusted_high_freq_time_free, high_freq_20_free['d']['I_SK_trafficking.S'], 'red', label = 'Sub. (5 Hz)', ls = 'dashed')
ax[1].axvline(x = 10, color = 'grey', ls = 'dotted', label = 'Protocol change')
ax[1].set_xlabel('Time [min]')
ax[1].set_ylabel('Number of channels')
ax[1].legend(loc = 'upper left')
ax[1].set_title('Free Ca2+')

plt.tight_layout()
plt.show()

#%% Run the reversibility protocol.

# Perform current clamp prepacing for 1 Hz.
pp_1hz_dem = current_prepace(pp = 1000, freq = 1, ca_i = ca_i)
pp_1hz_free = current_prepace(pp = 1000, freq = 1, ca_i = None)
pp_1hz_free_tblock = current_prepace(pp = 1000, freq = 1, ca_i = None, t_block = True)

# Perform the incubation with trafficking block. 
incub_free_tblock = incub_prot(pp = pp, ca_i = None, tblock = True)

# Run the reversibility protocol in voltage clamp.        
# if ca_i = 500 nM + voltage_clamp = incub_vcai
# if ca_i = None + voltage_clamp = incub_v
rev10_1hz_v_free = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = incub_free['incub_v'], current_clamp = False, ca_block = False)
rev20_5hz_v_free = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = rev10_1hz_v_free['state'], current_clamp = False, ca_block = False)
rev30_1hz_v_free = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = rev20_5hz_v_free['state'], current_clamp = False, ca_block = False)

# Additional plots for reversibility with small insets
rev11_5hz_v_free = reversibility(freq = 5, t = 0.1, interval = 1, ca_i = None, incub = rev10_1hz_v_free['state'], current_clamp = False, ca_block = False)
rev21_5hz_v_free = reversibility(freq = 1, t = 0.1, interval = 1, ca_i = None, incub = rev20_5hz_v_free['state'], current_clamp = False, ca_block = False)

# Run the reversibility protocol in current clamp. 
# if ca_i = None + current clamp = incub_comp
rev10_1hz_c_free = reversibility(freq = 1, t = 10, ca_i = None, interval = 5, incub = pp_1hz_free['incub_comp'], current_clamp = True)
rev20_5hz_c_free = reversibility(freq = 5, t = 10, ca_i = None, interval = 5, incub = rev10_1hz_c_free['state'], current_clamp = True)
rev30_1hz_c_free = reversibility(freq = 1, t = 10, ca_i = None, interval = 5, incub = rev20_5hz_c_free['state'], current_clamp = True)

# Additional plots for reversibility with small insets
rev11_5hz_c_free = reversibility(freq = 5, t = 0.1, ca_i = None, interval = 1, incub = rev10_1hz_c_free['state'], current_clamp = True)
rev21_5hz_c_free = reversibility(freq = 1, t = 0.1, interval = 1, ca_i = None, incub = rev20_5hz_c_free['state'], current_clamp = True)

# Re-run the simulations but block SK trafficking
rev10_1hz_v_free_tblock = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = incub_free_tblock['incub_v'], current_clamp = False, ca_block = False, t_block = True)
rev20_5hz_v_free_tblock = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = rev10_1hz_v_free_tblock['state'], current_clamp = False, ca_block = False, t_block = True)
rev30_1hz_v_free_tblock = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = rev20_5hz_v_free_tblock['state'], current_clamp = False, ca_block = False, t_block = True)

# Additional plots for reversibility with small insets
rev11_5hz_v_free_tblock = reversibility(freq = 5, t = 0.1, interval = 1, ca_i = None, incub = rev10_1hz_v_free_tblock['state'], current_clamp = False, ca_block = False, t_block = True)
rev21_5hz_v_free_tblock = reversibility(freq = 1, t = 0.1, interval = 1, ca_i = None, incub = rev20_5hz_v_free_tblock['state'], current_clamp = False, ca_block = False, t_block = True)

rev10_1hz_c_free_tblock = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub =  pp_1hz_free_tblock['incub_comp'], current_clamp = True, ca_block = False, t_block = True)
rev20_5hz_c_free_tblock = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = rev10_1hz_c_free_tblock['state'], current_clamp = True, ca_block = False, t_block = True)
rev30_1hz_c_free_tblock = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = rev20_5hz_c_free_tblock['state'], current_clamp = True, ca_block = False, t_block = True)

rev11_5hz_c_free_tblock = reversibility(freq = 5, t = 0.1, ca_i = None, interval = 1, incub = rev10_1hz_c_free_tblock['state'], current_clamp = True, ca_block = False, t_block = True)
rev21_5hz_c_free_tblock = reversibility(freq = 1, t = 0.1, interval = 1, ca_i = None, incub = rev20_5hz_c_free_tblock['state'], current_clamp = True, ca_block = False, t_block = True)

# Calculate the APD90s
first_10_free = calc_apd90(data = rev10_1hz_c_free, freq = 1, t = 10, cutoff = 1000)
second_10_free = calc_apd90(data = rev20_5hz_c_free, freq = 5, t = 10, cutoff = 1000)
third_10_free = calc_apd90(data = rev30_1hz_c_free, freq = 1, t = 10, cutoff = 1000) 

first_10_free_tblock = calc_apd90(data = rev10_1hz_c_free_tblock, freq = 1, t = 10, cutoff = 1000)
second_10_free_tblock = calc_apd90(data = rev20_5hz_c_free_tblock, freq = 5, t = 10, cutoff = 1000)
third_10_free_tblock = calc_apd90(data = rev30_1hz_c_free_tblock, freq = 1, t = 10, cutoff = 1000) 
#%% Reformat and export of reversibility simulations

# Reformat the simulation results
rev10_1hz_v_free_t = np.array(rev10_1hz_v_free['d']['engine.time'])/60000
rev20_5hz_v_free_t = np.array(rev20_5hz_v_free['d']['engine.time'])/60000
rev30_1hz_v_free_t = np.array(rev30_1hz_v_free['d']['engine.time'])/60000

rev10_1hz_c_free_t = np.array(rev10_1hz_c_free['d']['engine.time'])/60000
rev20_5hz_c_free_t = np.array(rev20_5hz_c_free['d']['engine.time'])/60000
rev30_1hz_c_free_t = np.array(rev30_1hz_c_free['d']['engine.time'])/60000

rev10_1hz_v_free_t_block = np.array(rev10_1hz_v_free_tblock['d']['engine.time'])/60000
rev20_5hz_v_free_t_block = np.array(rev20_5hz_v_free_tblock['d']['engine.time'])/60000
rev30_1hz_v_free_t_block = np.array(rev30_1hz_v_free_tblock['d']['engine.time'])/60000

rev10_1hz_c_free_t_block = np.array(rev10_1hz_c_free_tblock['d']['engine.time'])/60000
rev20_5hz_c_free_t_block = np.array(rev20_5hz_c_free_tblock['d']['engine.time'])/60000
rev30_1hz_c_free_t_block = np.array(rev30_1hz_c_free_tblock['d']['engine.time'])/60000

# Adjust the time from 10 and 20 minutes
rev20_5hz_v_free_ta = rev20_5hz_v_free_t + 10
rev30_1hz_v_free_ta = rev30_1hz_v_free_t + 20
rev20_5hz_c_free_ta = rev20_5hz_c_free_t + 10
rev30_1hz_c_free_ta = rev30_1hz_c_free_t + 20

rev20_5hz_v_free_ta_tblock = rev20_5hz_v_free_t_block + 10
rev30_1hz_v_free_ta_tblock = rev30_1hz_v_free_t_block + 20
rev20_5hz_c_free_ta_tblock = rev20_5hz_c_free_t_block + 10
rev30_1hz_c_free_ta_tblock = rev30_1hz_c_free_t_block + 20

# Reformat to dataframes
rev10_free1hz_df = pd.DataFrame({'Time_V': rev10_1hz_v_free_t[::5],
                                 'Membrane Channels': rev10_1hz_v_free['d']['I_SK_trafficking.M'][::5],
                                 'ISK': rev10_1hz_c_free['d']['I_SK.I_sk'][::5],
                                 'Time_C': rev10_1hz_c_free_t[::5],
                                 'Membrane potential': rev10_1hz_c_free['d']['membrane.V'][::5]})

rev9_free1hz_df_subset = rev10_free1hz_df[rev10_free1hz_df['Time_V'] >= 9.9]

rev20_free5hz_df = pd.DataFrame({'Time_V': rev20_5hz_v_free_ta[::5],
                                 'Membrane Channels': rev20_5hz_v_free['d']['I_SK_trafficking.M'][::5],
                                 'ISK': rev20_5hz_c_free['d']['I_SK.I_sk'][::5],
                                 'Time_C': rev20_5hz_c_free_ta[::5],
                                 'Membrane potential': rev20_5hz_c_free['d']['membrane.V'][::5]})

rev19_free_5hz_df_subset = rev20_free5hz_df[(rev20_free5hz_df['Time_V'] >= 19.9) & (rev20_free5hz_df['Time_V'] <= 20)]

rev30_free1hz_df = pd.DataFrame({'Time_V': rev30_1hz_v_free_ta[::5],
                                 'Membrane Channels': rev30_1hz_v_free['d']['I_SK_trafficking.M'][::5],
                                 'ISK': rev30_1hz_c_free['d']['I_SK.I_sk'][::5],
                                 'Time_C': rev30_1hz_c_free_ta[::5],
                                 'Membrane potential': rev30_1hz_c_free['d']['membrane.V'][::5]})

# Export the insets
rev11_inset_df = pd.DataFrame({'Time': np.array(rev11_5hz_v_free['d']['engine.time'])/60000 + 10,
                               'Membrane Channels': rev11_5hz_v_free['d']['I_SK_trafficking.M'],
                               'ISK': rev11_5hz_v_free['d']['I_SK.I_sk'],
                               'Membrane potential': rev11_5hz_c_free['d']['membrane.V'][:-1]})

rev21_inset_df = pd.DataFrame({'Time': np.array(rev21_5hz_v_free['d']['engine.time'])/60000 + 20,
                               'Membrane Channels': rev21_5hz_v_free['d']['I_SK_trafficking.M'],
                               'ISK': rev21_5hz_v_free['d']['I_SK.I_sk'],
                               'Membrane potential': rev21_5hz_c_free['d']['membrane.V'][:-1]})

rev11_inset_df_tblock = pd.DataFrame({'Time': np.array(rev11_5hz_v_free_tblock['d']['engine.time'])/60000 + 10,
                               'Membrane Channels': rev11_5hz_v_free_tblock['d']['I_SK_trafficking.M'],
                               'ISK': rev11_5hz_v_free_tblock['d']['I_SK.I_sk'],
                               'Membrane potential': rev11_5hz_c_free_tblock['d']['membrane.V'][:-1]})

rev21_inset_df_tblock = pd.DataFrame({'Time': np.array(rev21_5hz_v_free_tblock['d']['engine.time'])/60000 + 20,
                               'Membrane Channels': rev21_5hz_v_free_tblock['d']['I_SK_trafficking.M'],
                               'ISK': rev21_5hz_v_free_tblock['d']['I_SK.I_sk'],
                               'Membrane potential': rev21_5hz_c_free_tblock['d']['membrane.V'][:-1]})

# Repeat for the trafficking block
rev10_free1hz_tblock_df = pd.DataFrame({'Time_V': rev10_1hz_v_free_t_block[::5],
                                        'Membrane Channels': rev10_1hz_v_free_tblock['d']['I_SK_trafficking.M'][::5],
                                        'ISK': rev10_1hz_c_free_tblock['d']['I_SK.I_sk'][::5],
                                        'Time_C': rev10_1hz_c_free_t_block[::5],
                                        'Membrane potential': rev10_1hz_c_free_tblock['d']['membrane.V'][::5]})

rev9_free1hz_df_subset_tblock = rev10_free1hz_tblock_df[rev10_free1hz_tblock_df['Time_V'] >= 9.9]

rev20_free5hz_tblock_df = pd.DataFrame({'Time_V': rev20_5hz_v_free_ta_tblock[::5],
                                 'Membrane Channels': rev20_5hz_v_free_tblock['d']['I_SK_trafficking.M'][::5],
                                 'ISK': rev20_5hz_c_free_tblock['d']['I_SK.I_sk'][::5],
                                 'Time_C': rev20_5hz_c_free_ta_tblock[::5],
                                 'Membrane potential': rev20_5hz_c_free_tblock['d']['membrane.V'][::5]})

rev19_free_5hz_df_subset_tblock = rev20_free5hz_tblock_df[(rev20_free5hz_tblock_df['Time_V'] >= 19.9) & (rev20_free5hz_tblock_df['Time_V'] <= 20)]

rev30_free1hz_tblock_df = pd.DataFrame({'Time_V': rev30_1hz_v_free_ta_tblock[::5],
                                 'Membrane Channels': rev30_1hz_v_free_tblock['d']['I_SK_trafficking.M'][::5],
                                 'ISK': rev30_1hz_c_free_tblock['d']['I_SK.I_sk'][::5],
                                 'Time_C': rev30_1hz_c_free_ta_tblock[::5],
                                 'Membrane potential': rev30_1hz_c_free_tblock['d']['membrane.V'][::5]})

# Reformat the APD90 results
apd90_free_10 = pd.DataFrame({'Time': first_10_free['time'], 'APD90': first_10_free['apd90']})
apd90_free_20 = pd.DataFrame({'Time': second_10_free['time'][::20] + 10, 'APD90': second_10_free['apd90'][::20]})
apd90_free_30 = pd.DataFrame({'Time': third_10_free['time'][::20] + 20, 'APD90': third_10_free['apd90'][::20]})

apd_trafficblock_10 = pd.DataFrame({'Time': first_10_free_tblock['time'], 'APD':first_10_free_tblock['apd90']})
apd_trafficblock_20 =  pd.DataFrame({'Time': second_10_free_tblock['time'] + 10, 'APD':second_10_free_tblock['apd90']})
apd_trafficblock_30 =  pd.DataFrame({'Time': third_10_free_tblock['time'] + 20, 'APD':third_10_free_tblock['apd90']})

# Export the data to graphpad.
rev10_free1hz_df.to_csv('Results/rev10_free1hz.csv', index = False)
rev20_free5hz_df.to_csv('Results/rev20_free5hz.csv', index = False)
rev30_free1hz_df.to_csv('Results/rev30_free1hz.csv', index = False)

rev10_free1hz_tblock_df.to_csv('Results/rev10_free1hz_tblock.csv', index = False)
rev20_free5hz_tblock_df.to_csv('Results/rev20_free5hz_tblock.csv', index = False)
rev30_free1hz_tblock_df.to_csv('Results/rev30_free1hz_tblock.csv', index = False)

rev9_free1hz_df_subset.to_csv('Results/rev9_inset.csv', index = False)
rev11_inset_df.to_csv('Results/rev11_inset.csv', index = False)
rev19_free_5hz_df_subset.to_csv('Results/rev19_inset.csv', index = False)
rev21_inset_df.to_csv('Results/rev21_inset.csv', index = False)

rev9_free1hz_df_subset_tblock.to_csv('Results/rev9_inset_tblock.csv', index = False)
rev11_inset_df_tblock.to_csv('Results/rev11_inset_tblock.csv', index = False)
rev19_free_5hz_df_subset_tblock.to_csv('Results/rev19_inset_tblock.csv', index = False)
rev21_inset_df_tblock.to_csv('Results/rev21_inset_tblock.csv', index = False)

apd90_free_10.to_csv('Results/apd90_free_10.csv', index = False)
apd90_free_20.to_csv('Results/apd90_free_20.csv', index = False)
apd90_free_30.to_csv('Results/apd90_free_30.csv', index = False)

apd_trafficblock_10.to_csv('Results/apd90_tblock_10min.csv', index = False)
apd_trafficblock_20.to_csv('Results/apd90_tblock_20min.csv', index = False)
apd_trafficblock_30.to_csv('Results/apd90_tblock_30min.csv', index = False)

#%% APD90 plots for different pacing frequencies

# Perform current clamp prepacing for 1 Hz.
pp_1hz_free = current_prepace(pp = 1000, freq = 1, ca_i = None)
pp_1hz_free_tblock = current_prepace(pp = 1000, freq = 1, ca_i = None, t_block = True)

# Run the reversibility protocol in current clamp. 
# if ca_i = None + current clamp = incub_comp
apd90_1hz_free_10min = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free['incub_comp'], current_clamp = True)
apd90_2hz_free_10min = reversibility(freq = 2, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free['incub_comp'], current_clamp = True)
apd90_3hz_free_10min = reversibility(freq = 3, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free['incub_comp'], current_clamp = True)
apd90_4hz_free_10min = reversibility(freq = 4, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free['incub_comp'], current_clamp = True)
apd90_5hz_free_10min = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free['incub_comp'], current_clamp = True)

apd90_1hz_free_tblock_10min = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free_tblock['incub_comp'], current_clamp = True, t_block = True)
apd90_2hz_free_tblock_10min = reversibility(freq = 2, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free_tblock['incub_comp'], current_clamp = True, t_block = True)
apd90_3hz_free_tblock_10min = reversibility(freq = 3, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free_tblock['incub_comp'], current_clamp = True, t_block = True)
apd90_4hz_free_tblock_10min = reversibility(freq = 4, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free_tblock['incub_comp'], current_clamp = True, t_block = True)
apd90_5hz_free_tblock_10min = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = pp_1hz_free_tblock['incub_comp'], current_clamp = True, t_block = True)

# Calculate the APD90s
apd90_free_10min_1hz = calc_apd90(data = apd90_1hz_free_10min, freq = 1, t = 10, cutoff = 1000)
apd90_free_10min_2hz = calc_apd90(data = apd90_2hz_free_10min, freq = 2, t = 10, cutoff = 1000)
apd90_free_10min_3hz = calc_apd90(data = apd90_3hz_free_10min, freq = 3, t = 10, cutoff = 1000)
apd90_free_10min_4hz = calc_apd90(data = apd90_4hz_free_10min, freq = 4, t = 10, cutoff = 1000)
apd90_free_10min_5hz = calc_apd90(data = apd90_5hz_free_10min, freq = 5, t = 10, cutoff = 1000)

apd90_free_10min_1hz_tblock = calc_apd90(data = apd90_1hz_free_tblock_10min, freq = 1, t = 10, cutoff = 1000)
apd90_free_10min_2hz_tblock = calc_apd90(data = apd90_2hz_free_tblock_10min, freq = 2, t = 10, cutoff = 1000)
apd90_free_10min_3hz_tblock = calc_apd90(data = apd90_3hz_free_tblock_10min, freq = 3, t = 10, cutoff = 1000)
apd90_free_10min_4hz_tblock = calc_apd90(data = apd90_4hz_free_tblock_10min, freq = 4, t = 10, cutoff = 1000)
apd90_free_10min_5hz_tblock = calc_apd90(data = apd90_5hz_free_tblock_10min, freq = 5, t = 10, cutoff = 1000)

# Reformat for export purposes.
df_apd90_free_10min_1hz = pd.DataFrame({'Time': apd90_free_10min_1hz['time'], 'APD90': apd90_free_10min_1hz['apd90']})
df_apd90_free_10min_2hz = pd.DataFrame({'Time': apd90_free_10min_2hz['time'], 'APD90': apd90_free_10min_2hz['apd90']})
df_apd90_free_10min_3hz = pd.DataFrame({'Time': apd90_free_10min_3hz['time'], 'APD90': apd90_free_10min_3hz['apd90']})
df_apd90_free_10min_4hz = pd.DataFrame({'Time': apd90_free_10min_4hz['time'], 'APD90': apd90_free_10min_4hz['apd90']})
df_apd90_free_10min_5hz = pd.DataFrame({'Time': apd90_free_10min_5hz['time'], 'APD90': apd90_free_10min_5hz['apd90']})

df_apd90_free_10min_1hz_tblock = pd.DataFrame({'Time': apd90_free_10min_1hz_tblock['time'], 'APD90': apd90_free_10min_1hz_tblock['apd90']})
df_apd90_free_10min_2hz_tblock = pd.DataFrame({'Time': apd90_free_10min_2hz_tblock['time'], 'APD90': apd90_free_10min_2hz_tblock['apd90']})
df_apd90_free_10min_3hz_tblock = pd.DataFrame({'Time': apd90_free_10min_3hz_tblock['time'], 'APD90': apd90_free_10min_3hz_tblock['apd90']})
df_apd90_free_10min_4hz_tblock = pd.DataFrame({'Time': apd90_free_10min_4hz_tblock['time'], 'APD90': apd90_free_10min_4hz_tblock['apd90']})
df_apd90_free_10min_5hz_tblock = pd.DataFrame({'Time': apd90_free_10min_5hz_tblock['time'], 'APD90': apd90_free_10min_5hz_tblock['apd90']})

# Visualize the APD90 results.
plt.figure()
plt.plot(df_apd90_free_10min_1hz['Time'], df_apd90_free_10min_1hz['APD90'], label = '1 Hz')
plt.plot(df_apd90_free_10min_2hz['Time'], df_apd90_free_10min_2hz['APD90'], label = '2 Hz')
plt.plot(df_apd90_free_10min_3hz['Time'], df_apd90_free_10min_3hz['APD90'], label = '3 Hz')
plt.plot(df_apd90_free_10min_4hz['Time'], df_apd90_free_10min_4hz['APD90'], label = '4 Hz')
plt.plot(df_apd90_free_10min_5hz['Time'], df_apd90_free_10min_5hz['APD90'], label = '5 Hz')
plt.ylim([0, 500])
plt.legend()

plt.figure()
plt.plot(df_apd90_free_10min_1hz_tblock['Time'], df_apd90_free_10min_1hz_tblock['APD90'], label = '1 Hz')
plt.plot(df_apd90_free_10min_2hz_tblock['Time'], df_apd90_free_10min_2hz_tblock['APD90'], label = '2 Hz')
plt.plot(df_apd90_free_10min_3hz_tblock['Time'], df_apd90_free_10min_3hz_tblock['APD90'], label = '3 Hz')
plt.plot(df_apd90_free_10min_4hz_tblock['Time'], df_apd90_free_10min_4hz_tblock['APD90'], label = '4 Hz')
plt.plot(df_apd90_free_10min_5hz_tblock['Time'], df_apd90_free_10min_5hz_tblock['APD90'], label = '5 Hz')
plt.ylim([0, 500])
plt.title('block')
plt.legend()

# Export to graphpad.
df_apd90_free_10min_1hz.to_csv('Results/apd90_1hz_free_10.csv', index = False)
df_apd90_free_10min_2hz.to_csv('Results/apd90_2hz_free_10.csv', index = False)
df_apd90_free_10min_3hz.to_csv('Results/apd90_3hz_free_10.csv', index = False)
df_apd90_free_10min_4hz.to_csv('Results/apd90_4hz_free_10.csv', index = False)
df_apd90_free_10min_5hz .to_csv('Results/apd90_5hz_free_10.csv', index = False)

df_apd90_free_10min_1hz_tblock.to_csv('Results/apd90_1hz_free_10_tblock.csv', index = False)
df_apd90_free_10min_2hz_tblock.to_csv('Results/apd90_2hz_free_10_tblock.csv', index = False)
df_apd90_free_10min_3hz_tblock.to_csv('Results/apd90_3hz_free_10_tblock.csv', index = False)
df_apd90_free_10min_4hz_tblock.to_csv('Results/apd90_4hz_free_10_tblock.csv', index = False)
df_apd90_free_10min_5hz_tblock.to_csv('Results/apd90_5hz_free_10_tblock.csv', index = False)

#%% Linear sensitivity analysis

# Perform the sensitivity analysis by scaling the rates up and down. 
sens_arr = [3, 2, 1, 0.5, 0.33]
sens_arr_factor = [str(x) + 'x' for x in sens_arr]

# Initialize the list to hold scaled parameters
sens_list_alpha = []
sens_list_beta = []
sens_list_x = []

# Define the original parameter values
X = [0.05, 0.00035, 100, ]  # alpha, beta, x

# Step 1: Loop for scaling alpha, keep beta and x constant
for y in sens_arr:
    X_scale = X.copy()  # Make a copy of the original parameters
    X_scale[0] = X[0] * y  # Scale the alpha parameter
    sens_list_alpha.append(X_scale)  # Append to alpha scaling list

# Step 2: Loop for scaling beta, keep alpha and x constant
for y in sens_arr:
    X_scale = X.copy()  # Make a copy of the original parameters
    X_scale[1] = X[1] * y  # Scale the beta parameter
    sens_list_beta.append(X_scale)  # Append to beta scaling list

# Step 3: Loop for scaling x, keep alpha and beta constant
for y in sens_arr:
    X_scale = X.copy()  # Make a copy of the original parameters
    X_scale[2] = X[2] * y  # Scale the x parameter
    sens_list_x.append(X_scale)  # Append to x scaling list
    
def sens_analysis(x, scale, freq, t, incub, ca_i=None, param_name='alpha'):
    '''
    Function for sensitivity analysis by scaling one parameter at a time.
    '''

    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
     # Get the membrane component from the model.
    c = m.get('membrane')
    
    # Get the voltage variable and modify it
    v = c.get('V')
    v.set_binding(None)
    v.demote()
    
    # Get the level variable and modify it similarly.
    l = c.get('level')
    l.set_binding(None)
    
    if ca_i is not None:
        # Set intracellular Ca2+ concentration if provided.
        cai = m.get('Ca_Concentrations.Ca_i')
        cai.demote()
        cai.set_rhs(ca_i)
 
    # Define the basic cycle length.
    bcl = 1000 / freq
    
    # Add additional variables for pacing.
    v1 = c.add_variable('v1')
    v1.set_rhs('40 - 1.2 * (engine.time % '+ str(bcl) +')')
    
    vp = c.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')
    
    # Modify V equation for pacing.
    v.set_rhs('piecewise((engine.time % ' + str(bcl) + ' >= 0 and engine.time % ' + str(bcl) + ' <= 100), v1, vp)')
    
    # Initialize a protocol for the simulation.
    p = myokit.Protocol()
    
    # Calculate number of beats based on time and frequency.
    num_beats = int(t * freq * 60)
    
    for i in range(num_beats):
        p.add_step(-80, 100)
        p.add_step(-80, bcl - 100)
        
    # Total time of the simulation.
    t_trian = p.characteristic_time() - 1
    
    # Pre-allocate lists for results
    results = []
    
    # Define other simulation details (as per your original code)
    
    for i, scaling_set in enumerate(x):  # scaling_set is now a single list of scaled parameters
        s = myokit.Simulation(m, p)
        if freq < 5:
            s.set_max_step_size(0.05)
        else:
            s.set_max_step_size(0.02) 
        s.set_tolerance(1e-8, 1e-8)

        if ca_i is not None:
            s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        s.set_state(incub)
        
        # Set the parameters with their scaled values from `scaling_set`
        if param_name == 'alpha':
            s.set_constant('Ca_Concentrations.alpha', scaling_set[0])
        elif param_name == 'beta':
            s.set_constant('Ca_Concentrations.beta', scaling_set[1])
        elif param_name == 'x':
            s.set_constant('I_SK_trafficking.x', scaling_set[2])
        
        # Run the simulation and log results
        try:
            run = s.run(t_trian, log=['engine.time', 'I_SK_trafficking.M', 'I_SK_trafficking.S', 'Ca_Concentrations.fTrap'], log_interval=5)
        except myokit.SimulationError as e:
            print(f"Simulation failed at scaling index {i} with error: {e}")
            continue
        
        # Collect the results
        result = {
            'Time': run['engine.time'],
            'M': run['I_SK_trafficking.M'],
            'S': run['I_SK_trafficking.S'],
            'Steady_M': run['I_SK_trafficking.M'][-1],
            'Steady_S': run['I_SK_trafficking.S'][-1],
            'fTrap': run['Ca_Concentrations.fTrap'],
            'Scaling': scale[i]
        }
        results.append(result)

        s.reset()
    
    # Convert lists to DataFrames
    df_results = pd.DataFrame(results)
    
    return df_results

# Run the sensitivity analysis for alpha scaling
df_alpha_1hz = sens_analysis(x=sens_list_alpha, scale=sens_arr_factor, freq=1, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'alpha')
df_alpha_5hz = sens_analysis(x=sens_list_alpha, scale=sens_arr_factor, freq=5, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'alpha')
# Run the sensitivity analysis for beta scaling
df_beta_1hz = sens_analysis(x=sens_list_beta, scale=sens_arr_factor, freq=1, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'beta')
df_beta_5hz = sens_analysis(x=sens_list_beta, scale=sens_arr_factor, freq=5, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'beta')
# Run the sensitivity analysis for x scaling
df_x_1hz = sens_analysis(x=sens_list_x, scale=sens_arr_factor, freq=1, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'x')
df_x_5hz = sens_analysis(x=sens_list_x, scale=sens_arr_factor, freq=5, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'x')

# Initialize subplots with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True)

# Define titles for each subplot
titles = ['Alpha Scaling - 1 Hz', 'Alpha Scaling - 5 Hz', 
          'Beta Scaling - 1 Hz', 'Beta Scaling - 5 Hz', 
          'X Scaling - 1 Hz', 'X Scaling - 5 Hz']

# Define color palette
c_palette = ['black', 'blue', 'red', 'orange', 'purple']

# Plot for Alpha Scaling at 1 Hz (left column)
for i in range(len(df_alpha_1hz)):
    axs[0, 0].plot(np.array(df_alpha_1hz['Time'][i])/60000, np.array(df_alpha_1hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 0].set_title(titles[0])
axs[0, 0].set_ylabel('Membrane Channels')
axs[0, 0].legend()

# Plot for Alpha Scaling at 5 Hz (right column)
for i in range(len(df_alpha_5hz)):
    axs[0, 1].plot(np.array(df_alpha_5hz['Time'][i])/60000, np.array(df_alpha_5hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 1].set_title(titles[1])
axs[0, 1].set_ylabel('Membrane Channels')
axs[0, 1].legend()

# Plot for Beta Scaling at 1 Hz (left column)
for i in range(len(df_beta_1hz)):
    axs[1, 0].plot(np.array(df_beta_1hz['Time'][i])/60000, np.array(df_beta_1hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 0].set_title(titles[2])
axs[1, 0].set_ylabel('Membrane Channels')
axs[1, 0].legend()

# Plot for Beta Scaling at 5 Hz (right column)
for i in range(len(df_beta_5hz)):
    axs[1, 1].plot(np.array(df_beta_5hz['Time'][i])/60000, np.array(df_beta_5hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 1].set_title(titles[3])
axs[1, 1].set_ylabel('Membrane Channels')
axs[1, 1].legend()

# Plot for X Scaling at 1 Hz (left column)
for i in range(len(df_x_1hz)):
    axs[2, 0].plot(np.array(df_x_1hz['Time'][i])/60000, np.array(df_x_1hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[2, 0].set_title(titles[4])
axs[2, 0].set_xlabel('Time [min]')
axs[2, 0].set_ylabel('Membrane Channels')
axs[2, 0].legend()

# Plot for X Scaling at 5 Hz (right column)
for i in range(len(df_x_5hz)):
    axs[2, 1].plot(np.array(df_x_5hz['Time'][i])/60000, np.array(df_x_5hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[2, 1].set_title(titles[5])
axs[2, 1].set_xlabel('Time [min]')
axs[2, 1].set_ylabel('Membrane Channels')
axs[2, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Repeat for sub-membrane

# Initialize subplots with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True)

# Plot for Alpha Scaling at 1 Hz (left column)
for i in range(len(df_alpha_1hz)):
    axs[0, 0].plot(np.array(df_alpha_1hz['Time'][i])/60000, np.array(df_alpha_1hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 0].set_title(titles[0])
axs[0, 0].set_ylabel('Sub-membrane Channels')
axs[0, 0].legend()

# Plot for Alpha Scaling at 5 Hz (right column)
for i in range(len(df_alpha_5hz)):
    axs[0, 1].plot(np.array(df_alpha_5hz['Time'][i])/60000, np.array(df_alpha_5hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 1].set_title(titles[1])
axs[0, 1].set_ylabel('Sub-membrane Channels')
axs[0, 1].legend()

# Plot for Beta Scaling at 1 Hz (left column)
for i in range(len(df_beta_1hz)):
    axs[1, 0].plot(np.array(df_beta_1hz['Time'][i])/60000, np.array(df_beta_1hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 0].set_title(titles[2])
axs[1, 0].set_ylabel('Sub-membrane Channels')
axs[1, 0].legend()

# Plot for Beta Scaling at 5 Hz (right column)
for i in range(len(df_beta_5hz)):
    axs[1, 1].plot(np.array(df_beta_5hz['Time'][i])/60000, np.array(df_beta_5hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 1].set_title(titles[3])
axs[1, 1].set_ylabel('Sub-membrane Channels')
axs[1, 1].legend()

# Plot for X Scaling at 1 Hz (left column)
for i in range(len(df_x_1hz)):
    axs[2, 0].plot(np.array(df_x_1hz['Time'][i])/60000, np.array(df_x_1hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[2, 0].set_title(titles[4])
axs[2, 0].set_xlabel('Time [min]')
axs[2, 0].set_ylabel('Sub-embrane Channels')
axs[2, 0].legend()

# Plot for X Scaling at 5 Hz (right column)
for i in range(len(df_x_5hz)):
    axs[2, 1].plot(np.array(df_x_5hz['Time'][i])/60000, np.array(df_x_5hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[2, 1].set_title(titles[5])
axs[2, 1].set_xlabel('Time [min]')
axs[2, 1].set_ylabel('Sub-membrane Channels')
axs[2, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Make the frequency dependence plots. 
def sens_analysis_freq(df, sens_arr_factor):
    """
    Analyzes peak values in 'fTrap' for each scaling factor in a dataframe.
    
    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing time, fTrap, and scaling factor data for analysis.
        
    sens_arr_factor : list
        List of scaling factors to analyze, represented as strings (e.g., ['3x', '2x']).
    
    Returns:
    ----------
    peaks_by_scaling : dict
        Dictionary containing peak values, times, and indices for each scaling factor.
    """
    
    # Initialize a dictionary to store results for each scaling factor
    peaks_by_scaling = {}
    
    # Loop through each scaling factor in sens_arr_factor
    for scaling_factor in sens_arr_factor:
        # Subset the dataframe to include only rows with the current scaling factor
        subset = df[df['Scaling'] == scaling_factor]
        
        # Initialize lists to store peak values, times, and indices
        peak_vals = []    # List to hold peak values for current scaling factor
        peak_times = []   # List to hold times of each peak
        peak_idx = []     # List to hold indices of each peak within 'fTrap'
        
        # Loop through each row in the subset
        for idx, row in subset.iterrows():
            # Extract 'fTrap' list (values to find peaks in) and 'Time' list for the current row
            ftrap_values = row['fTrap']
            times = row['Time']
            
            # Ensure 'fTrap' is in a numpy array format for peak detection
            ftrap_values = np.array(ftrap_values)
            
            # Use find_peaks to detect indices of peaks in the 'fTrap' array
            peaks, _ = find_peaks(ftrap_values)
            
            # Store peak values and corresponding times by indexing with peak indices
            peak_vals.extend(ftrap_values[peaks].tolist())     # Append peak values to list
            peak_times.extend([times[i] for i in peaks])       # Append corresponding times of peaks
            peak_idx.extend(peaks.tolist())                    # Append peak indices
            
        # Store the peak values, times, and indices in the dictionary for the current scaling factor
        peaks_by_scaling[scaling_factor] = {
            'peak_vals': peak_vals,     # List of peak values for the scaling factor
            'peak_times': peak_times,   # List of times corresponding to each peak
            'peak_idx': peak_idx        # List of indices where peaks occur
        }
    
    # Return the dictionary with results for each scaling factor
    return peaks_by_scaling
# Get the frequency dependence plot (ftrap) for each pacing frequency
alpha_ftrap_1hz = sens_analysis_freq(df = df_alpha_1hz, sens_arr_factor = sens_arr_factor)
beta_ftrap_1hz = sens_analysis_freq(df = df_beta_1hz, sens_arr_factor = sens_arr_factor)
x_ftrap_1hz = sens_analysis_freq(df = df_x_1hz, sens_arr_factor = sens_arr_factor)

alpha_ftrap_5hz = sens_analysis_freq(df = df_alpha_5hz, sens_arr_factor = sens_arr_factor)
beta_ftrap_5hz = sens_analysis_freq(df = df_beta_5hz, sens_arr_factor = sens_arr_factor)
x_ftrap_5hz = sens_analysis_freq(df = df_x_5hz, sens_arr_factor = sens_arr_factor)


plt.figure()
for i in sens_arr_factor:
    plt.plot(np.array(alpha_ftrap_1hz[i]['peak_times'])/60000, alpha_ftrap_1hz[i]['peak_vals'], label = i)
plt.legend(loc = 'upper left')
plt.xlabel('Time [min]')
plt.ylabel('fTrap')
plt.title('Alpha [1 Hz]')

plt.figure()
for i in sens_arr_factor:
    plt.plot(np.array(beta_ftrap_1hz[i]['peak_times'])/60000, beta_ftrap_1hz[i]['peak_vals'], label = i)
plt.legend(loc = 'upper left')
plt.xlabel('Time [min]')
plt.ylabel('fTrap')
plt.title('Beta [1 Hz]')

plt.figure()
for i in sens_arr_factor:
    plt.plot(np.array(x_ftrap_1hz[i]['peak_times'])/60000, x_ftrap_1hz[i]['peak_vals'], label = i)
plt.legend(loc = 'upper left')
plt.xlabel('Time [min]')
plt.ylabel('fTrap')
plt.title('X [1 Hz]')

# Reformat for export purpose
alpha_scale_1hz = pd.DataFrame({'Time': np.array(df_alpha_1hz['Time'][0][::20])/60000, 
                                '3': df_alpha_1hz['M'][0][::20],
                                '2': df_alpha_1hz['M'][1][::20],
                                '1': df_alpha_1hz['M'][2][::20],
                                '0.5': df_alpha_1hz['M'][3][::20],
                                '0.33': df_alpha_1hz['M'][4][::20]})

alpha_scale_5hz = pd.DataFrame({'Time': np.array(df_alpha_5hz['Time'][0][::20])/60000, 
                                '3': df_alpha_5hz['M'][0][::20],
                                '2': df_alpha_5hz['M'][1][::20],
                                '1': df_alpha_5hz['M'][2][::20],
                                '0.5': df_alpha_5hz['M'][3][::20],
                                '0.33': df_alpha_5hz['M'][4][::20]})

beta_scale_1hz = pd.DataFrame({'Time': np.array(df_beta_1hz['Time'][0][::20])/60000, 
                                '3': df_beta_1hz['M'][0][::20],
                                '2': df_beta_1hz['M'][1][::20],
                                '1': df_beta_1hz['M'][2][::20],
                                '0.5': df_beta_1hz['M'][3][::20],
                                '0.33': df_beta_1hz['M'][4][::20]})

beta_scale_5hz = pd.DataFrame({'Time': np.array(df_beta_5hz['Time'][0][::20])/60000, 
                                '3': df_beta_5hz['M'][0][::20],
                                '2': df_beta_5hz['M'][1][::20],
                                '1': df_beta_5hz['M'][2][::20],
                                '0.5': df_beta_5hz['M'][3][::20],
                                '0.33': df_beta_5hz['M'][4][::20]})

x_scale_1hz = pd.DataFrame({'Time': np.array(df_x_1hz['Time'][0][::20])/60000, 
                                '3': df_x_1hz['M'][0][::20],
                                '2': df_x_1hz['M'][1][::20],
                                '1': df_x_1hz['M'][2][::20],
                                '0.5': df_x_1hz['M'][3][::20],
                                '0.33': df_x_1hz['M'][4][::20]})

x_scale_5hz = pd.DataFrame({'Time': np.array(df_x_5hz['Time'][0][::20])/60000, 
                                '3': df_x_5hz['M'][0][::20],
                                '2': df_x_5hz['M'][1][::20],
                                '1': df_x_5hz['M'][2][::20],
                                '0.5': df_x_5hz['M'][3][::20],
                                '0.33': df_x_5hz['M'][4][::20]})

# Repeat for the submembrane channels
alpha_scale_1hz_sub = pd.DataFrame({'Time': np.array(df_alpha_1hz['Time'][0][::20])/60000, 
                                    '3': df_alpha_1hz['S'][0][::20],
                                    '2': df_alpha_1hz['S'][1][::20],
                                    '1': df_alpha_1hz['S'][2][::20],
                                    '0.5': df_alpha_1hz['S'][3][::20],
                                    '0.33': df_alpha_1hz['S'][4][::20]})

alpha_scale_5hz_sub = pd.DataFrame({'Time': np.array(df_alpha_5hz['Time'][0][::20])/60000, 
                                    '3': df_alpha_5hz['S'][0][::20],
                                    '2': df_alpha_5hz['S'][1][::20],
                                    '1': df_alpha_5hz['S'][2][::20],
                                    '0.5': df_alpha_5hz['S'][3][::20],
                                    '0.33': df_alpha_5hz['S'][4][::20]})

beta_scale_1hz_sub = pd.DataFrame({'Time': np.array(df_beta_1hz['Time'][0][::20])/60000, 
                                   '3': df_beta_1hz['S'][0][::20],
                                   '2': df_beta_1hz['S'][1][::20],
                                   '1': df_beta_1hz['S'][2][::20],
                                   '0.5': df_beta_1hz['S'][3][::20],
                                   '0.33': df_beta_1hz['S'][4][::20]})

beta_scale_5hz_sub = pd.DataFrame({'Time': np.array(df_beta_5hz['Time'][0][::20])/60000, 
                                   '3': df_beta_5hz['S'][0][::20],
                                   '2': df_beta_5hz['S'][1][::20],
                                   '1': df_beta_5hz['S'][2][::20],
                                   '0.5': df_beta_5hz['S'][3][::20],
                                   '0.33': df_beta_5hz['S'][4][::20]})

x_scale_1hz_sub = pd.DataFrame({'Time': np.array(df_x_1hz['Time'][0][::20])/60000, 
                                '3': df_x_1hz['S'][0][::20],
                                '2': df_x_1hz['S'][1][::20],
                                '1': df_x_1hz['S'][2][::20],
                                '0.5': df_x_1hz['S'][3][::20],
                                '0.33': df_x_1hz['S'][4][::20]})

x_scale_5hz_sub = pd.DataFrame({'Time': np.array(df_x_5hz['Time'][0][::20])/60000, 
                                '3': df_x_5hz['S'][0][::20],
                                '2': df_x_5hz['S'][1][::20],
                                '1': df_x_5hz['S'][2][::20],
                                '0.5': df_x_5hz['S'][3][::20],
                                '0.33': df_x_5hz['S'][4][::20]})

# Put the dictionaries in a dataframe
df_alpha_ftrap_1hz_3x = pd.DataFrame({'Time': np.array(alpha_ftrap_1hz['3x']['peak_times'])/60000,
                                   '3': alpha_ftrap_1hz['3x']['peak_times']})

def export_sens_freq(d, sens_arr_factor, param, freq):
    """
    Exports peak values and times for different scaling factors to individual CSV files.

    Parameters:
    ----------
    d : dict
        Dictionary containing data for each scaling factor. For each factor, the dictionary should
        include 'peak_times' (time points where peaks occur) and 'peak_vals' (values at those peaks).
        
    sens_arr_factor : list of str
        List of scaling factors as strings, formatted with an "x" suffix (e.g., ['3x', '2x', '1x']).
        
    param : str
        The parameter being analyzed (e.g., 'alpha') to include in the filename.
        
    freq : int or float
        Frequency at which the data was collected, used in the filename.

    Returns:
    ----------
    None
        This function does not return any values. It generates CSV files with peak data for each scaling factor.
    """
    
    # Loop through each scaling factor in `sens_arr_factor`
    for factor in sens_arr_factor:

        # Create the DataFrame for the current scaling factor
        df = pd.DataFrame({
            'Time': np.array(d[factor]['peak_times']) / 60000,  # Convert peak times to minutes
            factor[:-1]: d[factor]['peak_vals']  # Column with peak values, using numeric part of factor as column name
        })
        
        # Export the DataFrame to a CSV file
        df.to_csv(f'Results/{param}_ftrap_{freq}hz_{factor}.csv', index=False)  # Save as CSV with custom filename


export_sens_freq(d = alpha_ftrap_1hz, sens_arr_factor = sens_arr_factor, param = 'alpha', freq = 1)
export_sens_freq(d = alpha_ftrap_5hz, sens_arr_factor = sens_arr_factor, param = 'alpha', freq = 5)
export_sens_freq(d = beta_ftrap_1hz, sens_arr_factor = sens_arr_factor, param = 'beta', freq = 1)
export_sens_freq(d = beta_ftrap_5hz, sens_arr_factor = sens_arr_factor, param = 'beta', freq = 5)
export_sens_freq(d = x_ftrap_1hz, sens_arr_factor = sens_arr_factor, param = 'x', freq = 1)
export_sens_freq(d = x_ftrap_5hz, sens_arr_factor = sens_arr_factor, param = 'x', freq = 5)

# Export to Graphpad
alpha_scale_1hz.to_csv('Results/alpha_scale_1hz.csv', index = False)
alpha_scale_5hz.to_csv('Results/alpha_scale_5hz.csv', index = False)
beta_scale_1hz.to_csv('Results/beta_scale_1hz.csv', index = False)
beta_scale_5hz.to_csv('Results/beta_scale_5hz.csv', index = False)
x_scale_1hz.to_csv('Results/x_scale_1hz.csv', index = False)
x_scale_5hz.to_csv('Results/x_scale_5hz.csv', index = False)

alpha_scale_1hz_sub.to_csv('Results/alpha_scale_1hz_sub.csv', index = False)
alpha_scale_5hz_sub.to_csv('Results/alpha_scale_5hz_sub.csv', index = False)
beta_scale_1hz_sub.to_csv('Results/beta_scale_1hz_sub.csv', index = False)
beta_scale_5hz_sub.to_csv('Results/beta_scale_5hz_sub.csv', index = False)
x_scale_1hz_sub.to_csv('Results/x_scale_1hz_sub.csv', index = False)
x_scale_5hz_sub.to_csv('Results/x_scale_5hz_sub.csv', index = False)


#%% Export the voltage/current clamp protocols

# Generate and plot the triangular protocol
triangular_data = voltage_clamp_prot(protocol_type="triangular", plot=True)

# Generate and plot the square pulse protocol as stacked pulses
square_data = voltage_clamp_prot(protocol_type="square", plot=True)

# Export the data to graphpad
triangular_data.to_csv('Results/triangular_protocol.csv', index = False)
square_data.to_csv('Results/square_protocol.csv', index = False)

#%% Graphical abstract

# First run the reversibility protocol for 10 minutes at 5 Hz to show the increase in membrane channels.
abs_1hz = reversibility(freq = 1, t = 10, interval = 5, ca_i = None, incub = rev10_1hz_v_free['state'], current_clamp = False, ca_block = False)
abs_5hz = reversibility(freq = 5, t = 10, interval = 5, ca_i = None, incub = rev10_1hz_v_free['state'], current_clamp = False, ca_block = False)

# Reformat into a dataframe
abs_5hz_df = pd.DataFrame({'Time_V': np.asarray(abs_5hz['d']['engine.time'][::100])/60000,
                           'Membrane Channels': abs_5hz['d']['I_SK_trafficking.M'][::100],
                           'ISK': abs_5hz['d']['I_SK.I_sk'][::100]})

abs_1hz_df = pd.DataFrame({'Time_V': np.asarray(abs_1hz['d']['engine.time'][::100])/60000,
                           'Membrane Channels': abs_1hz['d']['I_SK_trafficking.M'][::100],
                           'ISK': abs_1hz['d']['I_SK.I_sk'][::100]})

abs_5hz_1000_df = pd.DataFrame({'Time_V': np.asarray(abs_5hz['d']['engine.time'][::1000])/60000,
                           'Membrane Channels': abs_5hz['d']['I_SK_trafficking.M'][::1000],
                           'ISK': abs_5hz['d']['I_SK.I_sk'][::1000]})

abs_1hz_1000_df = pd.DataFrame({'Time_V': np.asarray(abs_1hz['d']['engine.time'][::1000])/60000,
                           'Membrane Channels': abs_1hz['d']['I_SK_trafficking.M'][::1000],
                           'ISK': abs_1hz['d']['I_SK.I_sk'][::1000]})


# Export to prism.
abs_1hz_df.to_csv('Results/abs_1hz_df.csv', index = False)
abs_5hz_df.to_csv('Results/abs_5hz_df.csv', index = False)
abs_1hz_1000_df.to_csv('Results/abs_1hz_1000_df.csv', index = False)
abs_5hz_1000_df.to_csv('Results/abs_5hz_1000_df.csv', index = False)

#%% Revision sensitivity analysis

# Perform the sensitivity analysis by scaling the rates up and down. 
sens_arr = [3, 2, 1, 0.5, 0.33]
sens_arr_factor = [str(x) + 'x' for x in sens_arr]

# Initialize the list to hold scaled parameters
sens_list_km = []
sens_list_hill = []

# Define the original parameter values 
X = [0.00115, 4]  # alpha, beta, x

# Step 1: Loop for scaling km, keep hill constant
for y in sens_arr:
    X_scale = X.copy()  # Make a copy of the original parameters
    X_scale[0] = X[0] * y  # Scale the km parameter
    sens_list_km.append(X_scale)  # Append to km scaling list

# Step 2: Loop for scaling hill, keep km constant
for y in sens_arr:
    X_scale = X.copy()  # Make a copy of the original parameters
    X_scale[1] = X[1] * y  # Scale the hill parameter
    sens_list_hill.append(X_scale)  # Append to hill scaling list


def sens_analysis_rev(x, scale, freq, t, incub, ca_i=None, param_name='km'):
    '''
    Function for sensitivity analysis by scaling one parameter at a time.
    '''

    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
     # Get the membrane component from the model.
    c = m.get('membrane')
    
    # Get the voltage variable and modify it
    v = c.get('V')
    v.set_binding(None)
    v.demote()
    
    # Get the level variable and modify it similarly.
    l = c.get('level')
    l.set_binding(None)
    
    if ca_i is not None:
        # Set intracellular Ca2+ concentration if provided.
        cai = m.get('Ca_Concentrations.Ca_i')
        cai.demote()
        cai.set_rhs(ca_i)
 
    # Define the basic cycle length.
    bcl = 1000 / freq
    
    # Add additional variables for pacing.
    v1 = c.add_variable('v1')
    v1.set_rhs('40 - 1.2 * (engine.time % '+ str(bcl) +')')
    
    vp = c.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')
    
    # Modify V equation for pacing.
    v.set_rhs('piecewise((engine.time % ' + str(bcl) + ' >= 0 and engine.time % ' + str(bcl) + ' <= 100), v1, vp)')
    
    # Initialize a protocol for the simulation.
    p = myokit.Protocol()
    
    # Calculate number of beats based on time and frequency.
    num_beats = int(t * freq * 60)
    
    for i in range(num_beats):
        p.add_step(-80, 100)
        p.add_step(-80, bcl - 100)
        
    # Total time of the simulation.
    t_trian = p.characteristic_time() - 1
    
    # Pre-allocate lists for results
    results = []
    
    # Define other simulation details (as per your original code)
    
    for i, scaling_set in enumerate(x):  # scaling_set is now a single list of scaled parameters
        s = myokit.Simulation(m, p)
        if freq < 5:
            s.set_max_step_size(0.05)
        else:
            s.set_max_step_size(0.02) 
        s.set_tolerance(1e-8, 1e-8)

        if ca_i is not None:
            s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        s.set_state(incub)
        
        # Set the parameters with their scaled values from `scaling_set`
        if param_name == 'km':
            s.set_constant('Ca_Concentrations.Km_Ca', scaling_set[0])
        elif param_name == 'hill':
            s.set_constant('Ca_Concentrations.hc', scaling_set[1])
        
        try:
            run = s.run(t_trian, log=['engine.time', 'I_SK_trafficking.M', 'I_SK_trafficking.S', 'Ca_Concentrations.fTrap'], log_interval=5)
        except myokit.SimulationError as e:
            print(f"Simulation failed at scaling index {i} with error: {e}")
            continue
        
        # Collect the results
        result = {
            'Time': run['engine.time'],
            'M': run['I_SK_trafficking.M'],
            'S': run['I_SK_trafficking.S'],
            'Steady_M': run['I_SK_trafficking.M'][-1],
            'Steady_S': run['I_SK_trafficking.S'][-1],
            'fTrap': run['Ca_Concentrations.fTrap'],
            'Scaling': scale[i]
        }
        results.append(result)

        s.reset()
    
    # Convert lists to DataFrames
    df_results = pd.DataFrame(results)
    
    return df_results

# Run the sensitivity analysis for km scaling
df_km_1hz = sens_analysis_rev(x=sens_list_km, scale=sens_arr_factor, freq=1, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'km')
df_km_5hz = sens_analysis_rev(x=sens_list_km, scale=sens_arr_factor, freq=5, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'km')

# Run the sensitivity analysis for hill scaling
df_hill_1hz = sens_analysis_rev(x=sens_list_hill, scale=sens_arr_factor, freq=1, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'hill')
df_hill_5hz = sens_analysis_rev(x=sens_list_hill, scale=sens_arr_factor, freq=5, t=10, incub=incub_free['incub_v'], ca_i=None, param_name = 'hill')

# Get the frequency dependence plot (ftrap) for each pacing frequency
km_ftrap_1hz = sens_analysis_freq(df = df_km_1hz, sens_arr_factor = sens_arr_factor)
hill_ftrap_1hz = sens_analysis_freq(df = df_hill_1hz, sens_arr_factor = sens_arr_factor)
km_ftrap_5hz = sens_analysis_freq(df = df_km_5hz, sens_arr_factor = sens_arr_factor)
hill_ftrap_5hz = sens_analysis_freq(df = df_hill_5hz, sens_arr_factor = sens_arr_factor)

#%% Visualize the revision sensitivity analysis plots.

# Initialize subplots with 3 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

# Define titles for each subplot
titles = ['Km Scaling - 1 Hz', 'Km Scaling - 5 Hz', 
          'Hill Scaling - 1 Hz', 'Hill Scaling - 5 Hz']

# Define color palette
c_palette = ['black', 'blue', 'red', 'orange', 'purple']

# Plot for km scaling at 1 Hz (left column)
for i in range(len(df_km_1hz)):
    axs[0, 0].plot(np.array(df_km_1hz['Time'][i])/60000, np.array(df_km_1hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 0].set_title(titles[0])
axs[0, 0].set_ylabel('Membrane Channels')
axs[0, 0].legend()

# Plot for km scaling at 5 Hz (right column)
for i in range(len(df_km_5hz)):
    axs[0, 1].plot(np.array(df_km_5hz['Time'][i])/60000, np.array(df_km_5hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 1].set_title(titles[1])
axs[0, 1].set_ylabel('Membrane Channels')
axs[0, 1].legend()

# Plot for hill scaling at 1 Hz (left column)
for i in range(len(df_hill_1hz)):
    axs[1, 0].plot(np.array(df_hill_1hz['Time'][i])/60000, np.array(df_hill_1hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 0].set_title(titles[2])
axs[1, 0].set_ylabel('Membrane Channels')
axs[1, 0].legend()

# Plot for hill scaling at 5 Hz (right column)
for i in range(len(df_hill_5hz)):
    axs[1, 1].plot(np.array(df_hill_5hz['Time'][i])/60000, np.array(df_hill_5hz['M'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 1].set_title(titles[3])
axs[1, 1].set_ylabel('Membrane Channels')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Repeat for sub-membrane

# Initialize subplots with 3 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)

# Plot for km scaling at 1 Hz (left column)
for i in range(len(df_km_1hz)):
    axs[0, 0].plot(np.array(df_km_1hz['Time'][i])/60000, np.array(df_km_1hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 0].set_title(titles[0])
axs[0, 0].set_ylabel('Sub-membrane Channels')
axs[0, 0].legend()

# Plot for km scaling at 5 Hz (right column)
for i in range(len(df_km_5hz)):
    axs[0, 1].plot(np.array(df_km_5hz['Time'][i])/60000, np.array(df_km_5hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[0, 1].set_title(titles[1])
axs[0, 1].set_ylabel('Sub-membrane Channels')
axs[0, 1].legend()

# Plot for hill scaling at 1 Hz (left column)
for i in range(len(df_hill_1hz)):
    axs[1, 0].plot(np.array(df_hill_1hz['Time'][i])/60000, np.array(df_hill_1hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 0].set_title(titles[2])
axs[1, 0].set_ylabel('Sub-membrane Channels')
axs[1, 0].legend()

# Plot for hill scaling at 5 Hz (right column)
for i in range(len(df_hill_5hz)):
    axs[1, 1].plot(np.array(df_hill_5hz['Time'][i])/60000, np.array(df_hill_5hz['S'][i]), color=c_palette[i], label=sens_arr_factor[i])
axs[1, 1].set_title(titles[3])
axs[1, 1].set_ylabel('Sub-membrane Channels')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Reformat for export purpose
km_scale_1hz = pd.DataFrame({'Time': np.array(df_km_1hz['Time'][0][::20])/60000, 
                                '3': df_km_1hz['M'][0][::20],
                                '2': df_km_1hz['M'][1][::20],
                                '1': df_km_1hz['M'][2][::20],
                                '0.5': df_km_1hz['M'][3][::20],
                                '0.33': df_km_1hz['M'][4][::20]})

km_scale_5hz = pd.DataFrame({'Time': np.array(df_km_5hz['Time'][0][::20])/60000, 
                                '3': df_km_5hz['M'][0][::20],
                                '2': df_km_5hz['M'][1][::20],
                                '1': df_km_5hz['M'][2][::20],
                                '0.5': df_km_5hz['M'][3][::20],
                                '0.33': df_km_5hz['M'][4][::20]})

hill_scale_1hz = pd.DataFrame({'Time': np.array(df_hill_1hz['Time'][0][::20])/60000, 
                                '3': df_hill_1hz['M'][0][::20],
                                '2': df_hill_1hz['M'][1][::20],
                                '1': df_hill_1hz['M'][2][::20],
                                '0.5': df_hill_1hz['M'][3][::20],
                                '0.33': df_hill_1hz['M'][4][::20]})

hill_scale_5hz = pd.DataFrame({'Time': np.array(df_hill_5hz['Time'][0][::20])/60000, 
                                '3': df_hill_5hz['M'][0][::20],
                                '2': df_hill_5hz['M'][1][::20],
                                '1': df_hill_5hz['M'][2][::20],
                                '0.5': df_hill_5hz['M'][3][::20],
                                '0.33': df_hill_5hz['M'][4][::20]})


# Repeat for the submembrane channels
km_scale_1hz_sub = pd.DataFrame({'Time': np.array(df_km_1hz['Time'][0][::20])/60000, 
                                    '3': df_km_1hz['S'][0][::20],
                                    '2': df_km_1hz['S'][1][::20],
                                    '1': df_km_1hz['S'][2][::20],
                                    '0.5': df_km_1hz['S'][3][::20],
                                    '0.33': df_km_1hz['S'][4][::20]})

km_scale_5hz_sub = pd.DataFrame({'Time': np.array(df_km_5hz['Time'][0][::20])/60000, 
                                    '3': df_km_5hz['S'][0][::20],
                                    '2': df_km_5hz['S'][1][::20],
                                    '1': df_km_5hz['S'][2][::20],
                                    '0.5': df_km_5hz['S'][3][::20],
                                    '0.33': df_km_5hz['S'][4][::20]})

hill_scale_1hz_sub = pd.DataFrame({'Time': np.array(df_hill_1hz['Time'][0][::20])/60000, 
                                   '3': df_hill_1hz['S'][0][::20],
                                   '2': df_hill_1hz['S'][1][::20],
                                   '1': df_hill_1hz['S'][2][::20],
                                   '0.5': df_hill_1hz['S'][3][::20],
                                   '0.33': df_hill_1hz['S'][4][::20]})

hill_scale_5hz_sub = pd.DataFrame({'Time': np.array(df_hill_5hz['Time'][0][::20])/60000, 
                                   '3': df_hill_5hz['S'][0][::20],
                                   '2': df_hill_5hz['S'][1][::20],
                                   '1': df_hill_5hz['S'][2][::20],
                                   '0.5': df_hill_5hz['S'][3][::20],
                                   '0.33': df_hill_5hz['S'][4][::20]})

# Export to Graphpad
km_scale_1hz.to_csv('Results/km_scale_1hz.csv', index = False)
km_scale_5hz.to_csv('Results/km_scale_5hz.csv', index = False)
hill_scale_1hz.to_csv('Results/hill_scale_1hz.csv', index = False)
hill_scale_5hz.to_csv('Results/hill_scale_5hz.csv', index = False)

km_scale_1hz_sub.to_csv('Results/km_scale_1hz_sub.csv', index = False)
km_scale_5hz_sub.to_csv('Results/km_scale_5hz_sub.csv', index = False)
hill_scale_1hz_sub.to_csv('Results/hill_scale_1hz_sub.csv', index = False)
hill_scale_5hz_sub.to_csv('Results/hill_scale_5hz_sub.csv', index = False)

export_sens_freq(d = km_ftrap_1hz, sens_arr_factor = sens_arr_factor, param = 'km', freq = 1)
export_sens_freq(d = km_ftrap_5hz, sens_arr_factor = sens_arr_factor, param = 'km', freq = 5)
export_sens_freq(d = hill_ftrap_1hz, sens_arr_factor = sens_arr_factor, param = 'hill', freq = 1)
export_sens_freq(d = hill_ftrap_5hz, sens_arr_factor = sens_arr_factor, param = 'hill', freq = 5)

#%% Population of models approach

# Load the model
m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

# Define the parameters
param_names = ['Ca_Concentrations.alpha', 'Ca_Concentrations.beta', 'Ca_Concentrations.Ca_basal_act',
               'Ca_Concentrations.Km_Ca', 'Ca_Concentrations.hc', 'I_SK_trafficking.a', 'I_SK_trafficking.b',
               'I_SK_trafficking.d', 'I_SK_trafficking.p', 'I_SK_trafficking.x']

# Get the current values of parameters from the model
param_vals = [m.get(p).value() for p in param_names]

# Number of populations 
n_sims = 100

# Number of repeats
n_repeats = 5

# Generate the populations, note this takes a long time.
# PoM_res = PoM(m = m, freq1 = 1, freq2 = 5, t = 10, incub = pp_1hz_free['incub_comp'], interval = 5, 
#               param_names = param_names, param_vals = param_vals, n_sims = n_sims, n_repeats = n_repeats, seed = 42)

# Export the results
# export_PoM_results_by_repeat(PoM_res, base_name='PoM')

#There has been a slight mistake in the concatenation which will be fixed by the following function
# correct_time_glitch('Results/PoM_trafficking_M_repeat1.csv')
# correct_time_glitch('Results/PoM_trafficking_M_repeat2.csv')
# correct_time_glitch('Results/PoM_trafficking_M_repeat3.csv')
# correct_time_glitch('Results/PoM_trafficking_M_repeat4.csv')
# correct_time_glitch('Results/PoM_trafficking_M_repeat5.csv')

#%% PLS analysis Membrane Channels

# First perform the PLS analysis for the membrane channels
sens_df_10min = []  # For 10 min
sens_df_20min = []  # For 20 min

for j in range(1, n_repeats + 1):
    # Load parameters and channel data
    params = pd.read_csv(f'Results/PoM_params_repeat{j}.csv')
    mem = pd.read_csv(f'Results/PoM_trafficking_M_repeat{j}.csv')

    # Unified sensitivity function using 'channel' mode
    sens_10 = PoM_sens_PLS(mem, params, param_names, interval=5, t=10, mode='channel')
    sens_20 = PoM_sens_PLS(mem, params, param_names, interval=5, t=20, mode='channel')

    sens_df_10min.append(sens_10)
    sens_df_20min.append(sens_20)

# Stack and summarize
sens_10_array = np.stack([df['Sensitivity'].values for df in sens_df_10min])
sens_20_array = np.stack([df['Sensitivity'].values for df in sens_df_20min])

mean_10, std_10 = sens_10_array.mean(axis=0), sens_10_array.std(axis=0)
mean_20, std_20 = sens_20_array.mean(axis=0), sens_20_array.std(axis=0)

# Plot
x = np.arange(len(param_names))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, mean_10, width, yerr=std_10, capsize=5, label='Channels (10 min)', color='teal')
plt.bar(x + width/2, mean_20, width, yerr=std_20, capsize=5, label='Channels (20 min)', color='darkorange')
plt.xticks(x, param_names, rotation=45, ha='right')
plt.ylabel('PLS Sensitivity (Standardized)')
plt.title('Channel Sensitivity via PLS (Mean ± SD across repeats)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Create lists for each timepoint
df_10_mem = pd.DataFrame({'Parameter': param_names, 'Mean': mean_10, 'StdDev': std_10})
df_20_mem = pd.DataFrame({'Parameter': param_names,'Mean': mean_20,'StdDev': std_20})

df_10_mem.to_csv('Results/pls_mem10.csv', index=False)
df_20_mem.to_csv('Results/pls_mem20.csv', index = False)

# Create DataFrame in wide format
graphpad_df = pd.DataFrame({'Parameter': param_names,'10 min Mean': mean_10,'10 min SD': std_10,'20 min Mean': mean_20,'20 min SD': std_20})

# Export to CSV
graphpad_df.to_csv('Results/pls_sensitivity_membrane.csv', index=False)
#%% PLS analysis APD90

# Repeat the PLS for the APD90
sens_df_apd10 = []
sens_df_apd20 = []

for j in range(1, n_repeats + 1):
    apd = pd.read_csv(f'Results/PoM_APDs_repeat{j}.csv')
    apd['APD_list'] = apd['APD'].apply(ast.literal_eval)
    apd['apd90_10min'] = apd['APD_list'].apply(lambda x: x[599] if len(x) > 599 else None)
    apd['apd90_20min'] = apd['APD_list'].apply(lambda x: x[-1] if len(x) > 0 else None)

    params = pd.read_csv(f'Results/PoM_params_repeat{j}.csv')

    # Use unified function in 'apd90' mode
    sens_10 = PoM_sens_PLS(apd['apd90_10min'], params, param_names, mode='apd90')
    sens_20 = PoM_sens_PLS(apd['apd90_20min'], params, param_names, mode='apd90')

    sens_df_apd10.append(sens_10)
    sens_df_apd20.append(sens_20)

# Stack and summarize
sens_10_array_apd = np.stack([df['Sensitivity'].values for df in sens_df_apd10])
sens_20_array_apd = np.stack([df['Sensitivity'].values for df in sens_df_apd20])

mean_apd_10, std_apd_10 = sens_10_array_apd.mean(axis=0), sens_10_array_apd.std(axis=0)
mean_apd_20, std_apd_20 = sens_20_array_apd.mean(axis=0), sens_20_array_apd.std(axis=0)

# Plot
x = np.arange(len(param_names))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, mean_apd_10, width, yerr=std_apd_10, capsize=5, label='APD90 (10 min)', color='royalblue')
plt.bar(x + width/2, mean_apd_20, width, yerr=std_apd_20, capsize=5, label='APD90 (20 min)', color='orange')
plt.xticks(x, param_names, rotation=45, ha='right')
plt.ylabel('PLS Sensitivity (Standardized)')
plt.title('APD90 Sensitivity via PLS (Mean ± SD across repeats)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Visualize the raw PoM results.
visualize_PoM_outputs(output_type='channel')        # Channel time courses
visualize_PoM_outputs(output_type='vm_10min')       # Vm beat at 10 min
visualize_PoM_outputs(output_type='vm_20min')       # Vm beat at 20 min


# Create DataFrame in wide format
apd_sens_df = pd.DataFrame({'Parameter': param_names,'10 min Mean': mean_apd_10,'10 min SD': std_apd_10,'20 min Mean': mean_apd_20,'20 min SD': std_apd_20})

# Export to CSV
apd_sens_df.to_csv('Results/pls_sensitivity_apd.csv', index=False)
#%% IKuR and IKr block

# Time in ms.
ten_min_ms = 600000
twenty_min_ms = 1200000

# Perform pre-pacing for the drug effects
pp_1hz_skb = current_prepace_drugs(pp = 1000, AVE0118_conc = 10, freq = 1, sk_block = True, ikur_block = False, ikr_block = False, k2p_block = False, combined = False)
pp_1hz_ikurb = current_prepace_drugs(pp = 1000, AVE0118_conc = 10, freq = 1, sk_block = False, ikur_block = True, ikr_block = False, k2p_block = False, combined = False)
pp_1hz_ikrb = current_prepace_drugs(pp = 1000, AVE0118_conc = 10, freq = 1, sk_block = False, ikur_block = False, ikr_block = True, k2p_block = False, combined = False)
pp_1hz_k2pb = current_prepace_drugs(pp = 1000, AVE0118_conc = 10, freq = 1, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = True, combined = False)
pp_1hz_comb = current_prepace_drugs(pp = 1000, AVE0118_conc = 10, freq = 1, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = False, combined = True)

# Run the simulation at 1 Hz and 5 Hz 
sk_block_1hz_10min = drug_effects(freq = 1, t = 10, interval = 1, incub = pp_1hz_skb['incub_comp'], AVE0118_conc = 10, sk_block = True, ikur_block = False, ikr_block = False, k2p_block = False, combined = False)
sk_block_5hz_20min = drug_effects(freq = 5, t = 10, interval = 1, incub = sk_block_1hz_10min['state'], AVE0118_conc = 10, sk_block = True, ikur_block = False, ikr_block = False, k2p_block = False, combined = False)

ikur_block_1hz_10min = drug_effects(freq = 1, t = 10, interval = 1, incub = pp_1hz_ikurb['incub_comp'], AVE0118_conc = 10, sk_block = False, ikur_block = True, ikr_block = False, k2p_block = False, combined = False)
ikur_block_5hz_20min = drug_effects(freq = 5, t = 10, interval = 1, incub = ikur_block_1hz_10min['state'], AVE0118_conc = 10, sk_block = False, ikur_block = True, ikr_block = False, k2p_block = False, combined = False)

ikr_block_1hz_10min = drug_effects(freq = 1, t = 10, interval = 1, incub = pp_1hz_ikrb['incub_comp'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = True, k2p_block = False, combined = False)
ikr_block_5hz_20min = drug_effects(freq = 5, t = 10, interval = 1, incub = ikr_block_1hz_10min['state'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = True, k2p_block = False, combined = False)

k2p_block_1hz_10min = drug_effects(freq = 1, t = 10, interval = 1, incub = pp_1hz_k2pb['incub_comp'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = True, combined = False)
k2p_block_5hz_20min = drug_effects(freq = 5, t = 10, interval = 1, incub = k2p_block_1hz_10min['state'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = True, combined = False)

comb_block_1hz_10min = drug_effects(freq = 1, t = 10, interval = 1, incub = pp_1hz_comb['incub_comp'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = False, combined = True)
comb_block_5hz_20min = drug_effects(freq = 5, t = 10, interval = 1, incub = comb_block_1hz_10min['state'], AVE0118_conc = 10, sk_block = False, ikur_block = False, ikr_block = False, k2p_block = False, combined = True)

# Get the insets
rev11_5hz_c_free = reversibility(freq = 5, t = 0.1, ca_i = None, interval = 1, incub = rev10_1hz_c_free['state'], current_clamp = True)
rev21_5hz_c_free = reversibility(freq = 1, t = 0.1, interval = 1, ca_i = None, incub = rev20_5hz_c_free['state'], current_clamp = True)

# Calculate the APD90
skb_10min_apd = calc_apd90(data = sk_block_1hz_10min, freq = 1, t = 10, cutoff = 1000)
skb_20min_apd = calc_apd90(data = sk_block_5hz_20min, freq = 5, t = 10, cutoff = 1000)

ikurb_10min_apd = calc_apd90(data = ikur_block_1hz_10min, freq = 1, t = 10, cutoff = 1000)
ikurb_20min_apd = calc_apd90(data = ikur_block_5hz_20min, freq = 5, t = 10, cutoff = 1000)

ikrb_10min_apd = calc_apd90(data = ikr_block_1hz_10min, freq = 1, t = 10, cutoff = 1000)
ikrb_20min_apd = calc_apd90(data = ikr_block_5hz_20min, freq = 5, t = 10, cutoff = 1000)

k2p_10min_apd = calc_apd90(data = k2p_block_1hz_10min, freq = 1, t = 10, cutoff = 1000)
k2p_20min_apd = calc_apd90(data = k2p_block_5hz_20min, freq = 5, t = 10, cutoff = 1000)

comb_10min_apd = calc_apd90(data = comb_block_1hz_10min, freq = 1, t = 10, cutoff = 1000)
comb_20min_apd = calc_apd90(data = comb_block_5hz_20min, freq = 5, t = 10, cutoff = 1000)

# Create dataframes for each scenario
baseline_apd_df = create_block_df_apd(first_10_free, second_10_free)
sk_block_df_apd = create_block_df_apd(skb_10min_apd, skb_20min_apd)
ikur_block_df_apd = create_block_df_apd(ikurb_10min_apd, ikurb_20min_apd)
ikr_block_df_apd = create_block_df_apd(ikrb_10min_apd, ikrb_20min_apd)
k2p_block_df_apd = create_block_df_apd(k2p_10min_apd, k2p_20min_apd)
comb_block_df_apd = create_block_df_apd(comb_10min_apd, comb_20min_apd)

baseline_df = create_block_df(rev10_1hz_c_free, rev20_5hz_c_free, offset = ten_min_ms)
sk_block_df = create_block_df(sk_block_1hz_10min, sk_block_5hz_20min, offset = ten_min_ms)
ikur_block_df = create_block_df(ikur_block_1hz_10min, ikur_block_5hz_20min, offset = ten_min_ms)
ikr_block_df = create_block_df(ikr_block_1hz_10min , ikr_block_5hz_20min, offset = ten_min_ms)
k2p_block_df = create_block_df(k2p_block_1hz_10min, k2p_block_5hz_20min, offset = ten_min_ms)
comb_block_df = create_block_df(comb_block_1hz_10min , comb_block_5hz_20min, offset = ten_min_ms)

def segment_extract(df, t10, t20, c1, c2):
    segment_10 = df[(df['Time'] >= (t10 - c1)) & (df['Time'] < t10)].copy()
    segment_10['Aligned_Time'] = segment_10['Time'] - (t10 - c1)
    
    segment_20 = df[(df['Time'] >= (t20 - c2)) & (df['Time'] < t20)].copy()
    segment_20['Aligned_Time'] = segment_20['Time'] - (t20 - c2)
    
    return segment_10, segment_20

base_10, base_20 = segment_extract(df = baseline_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)
skb_10, skb_20 = segment_extract(df = sk_block_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)
ikur_10, ikur_20 = segment_extract(df = ikur_block_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)
ikr_10, ikr_20 = segment_extract(df = ikr_block_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)
k2p_10, k2p_20 = segment_extract(df = k2p_block_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)
comb_10, comb_20 = segment_extract(df = comb_block_df, t10 = ten_min_ms, t20 = twenty_min_ms, c1 = 1000, c2 = 1000)


plt.figure()
plt.plot(base_10['Aligned_Time'], base_10['Vm'], label = f"No Block, APD90 = {round(first_10_free['apd90'][-1])} ms")
plt.plot(skb_10['Aligned_Time'], skb_10['Vm'], label = f"SK block, APD90 = {round(skb_10min_apd['apd90'][-1])} ms")
plt.plot(ikur_10['Aligned_Time'], ikur_10['Vm'], label = f"IKur block, APD90 = {round(ikurb_10min_apd['apd90'][-1])} ms")
plt.plot(ikr_10['Aligned_Time'], ikr_10['Vm'], label = f"IKr block, APD90 = {round(ikrb_10min_apd['apd90'][-1])} ms")
plt.plot(k2p_10['Aligned_Time'], k2p_10['Vm'], label = f"K2P block, APD90 = {round(k2p_10min_apd['apd90'][-1])} ms")
plt.plot(comb_10['Aligned_Time'], comb_10['Vm'], label = f"K2P + SK block, APD90 = {round(comb_10min_apd['apd90'][-1])} ms")
plt.legend()
plt.title('1 Hz')


plt.figure()
plt.plot(base_20['Aligned_Time'], base_20['Vm'], label = f"No Block, APD90 = {round(second_10_free['apd90'][-1])} ms")
plt.plot(skb_20['Aligned_Time'], skb_20['Vm'], label = f"SK block, APD90 = {round(skb_20min_apd['apd90'][-1])} ms")
plt.plot(ikur_20['Aligned_Time'], ikur_20['Vm'], label = f"IKur block, APD90 = {round(ikurb_20min_apd['apd90'][-1])} ms")
plt.plot(ikr_20['Aligned_Time'], ikr_20['Vm'], label = f"IKr block, APD90 = {round(ikrb_20min_apd['apd90'][-1])} ms")
plt.plot(k2p_20['Aligned_Time'], k2p_20['Vm'], label = f"K2P block, APD90 = {round(k2p_20min_apd['apd90'][-1])} ms")
plt.plot(comb_20['Aligned_Time'], comb_20['Vm'], label = f"K2P + SK block, APD90 = {round(comb_20min_apd['apd90'][-1])} ms")
plt.legend()
plt.title('5 Hz')

#%% Export the dataframes.
base_10.to_csv("Results/baseline_10min_segment.csv", index=False)
base_20.to_csv("Results/baseline_20min_segment.csv", index=False)

skb_10.to_csv("Results/skblock_10min_segment.csv", index=False)
skb_20.to_csv("Results/skblock_20min_segment.csv", index=False)

ikur_10.to_csv("Results/ikurblock_10min_segment.csv", index=False)
ikur_20.to_csv("Results/ikurblock_20min_segment.csv", index=False)

ikr_10.to_csv("Results/ikrblock_10min_segment.csv", index=False)
ikr_20.to_csv("Results/ikrblock_20min_segment.csv", index=False)

k2p_10.to_csv("Results/ik2pblock_10min_segment.csv", index=False)
k2p_20.to_csv("Results/ik2pblock_20min_segment.csv", index=False)

comb_10.to_csv("Results/combblock_10min_segment.csv", index=False)
comb_20.to_csv("Results/combblock_20min_segment.csv", index=False)
