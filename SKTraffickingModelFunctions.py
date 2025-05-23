#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Stefan Meier
Institute: CARIM Maastricht University
Supervisor: Prof. Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 03/11/2023
Function: Optimization functions 
"""
# Import the relevant packages
import numpy as np
import pandas as pd 
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.signal import find_peaks
from statistics import median 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
import myokit
import os

#%%

def add_beat(protocol, vhold, thold, vdep, tdep, tdec, steps, beat_type='triangular'):
    """
    Adds a beat to the Myokit protocol with specified parameters.
    
    Parameters:
    ----------
    protocol : myokit.Protocol
        The Myokit Protocol object to which the beat is added.
    
    vhold : float
        The holding potential in mV.
    
    thold : float
        The duration of the holding step in ms.
    
    vdep : float
        The depolarizing or step potential in mV.
    
    tdep : float
        The duration of the depolarizing or step in ms.
    
    tdec : float
        The duration of the decay phase or return to holding potential in ms.
    
    steps : int
        The number of steps for the decay (used only for triangular beat).
    
    beat_type : str
        The type of beat to add ('triangular' or 'square').
    
    Returns:
    ----------
    None
        This function modifies the protocol in place and does not return any value.
    """
    if beat_type == 'triangular':
        # Add the initial holding step.
        protocol.add_step(vhold, thold)
        
        # Add the depolarizing step.
        protocol.add_step(vdep, tdep)
        
        # Calculate the voltage decrement for each step.
        delta_v = (vhold - vdep) / steps
        
        # Add the decaying steps.
        for i in range(steps):
            voltage = vdep + i * delta_v
            protocol.add_step(voltage, tdec / steps)
    
    elif beat_type == 'square':
        # Add the initial holding step
        protocol.add_step(vhold, thold)
        
        # Add the step
        protocol.add_step(vdep, tdep)
        
        # Return to holding potential
        protocol.add_step(vhold, thold)
    
    else:
        raise ValueError("Invalid beat_type. Use 'triangular' or 'square'.")

def time_freq(frequencies, vhold, vdep, tdec, steps, beat_type):
    """
    Calculate and print the holding and depolarizing durations for various frequencies.

    Parameters:
    ----------
    frequencies : list
        List of frequencies (in Hz) for which to calculate the durations.
    vhold : int
        Holding potential (in mV).
    vdep : int
        Depolarizing voltage (in mV).
    tdec : int
        Decay duration (in ms) for triangular beat.
    steps : int
        Number of steps for the decay.
    beat_type : str
        Type of beat ('square' or 'triangular').

    Returns:
    ----------
    results_df : pd.DataFrame
        A DataFrame containing the frequency, depolarizing duration (tdep), and holding durations (thold1 and thold2 or thold)
        for each frequency.
    """
    results = []

    for freq in frequencies:
        # Calculate the total duration for one beat in ms.
        total_duration = 1000 / freq

        if beat_type == 'square':
            # Calculate holding durations and depolarizing duration for square protocol.
            thold1 = thold2 = (total_duration - tdec) / 2
            tdep = total_duration - thold1 - thold2
            # Ensure the calculated depolarizing duration is valid.
            if tdep < 0:
                raise ValueError("Invalid protocol: Depolarizing duration is negative.")
        else:
            # Calculate holding durations and depolarizing duration for triangular protocol.
            thold1 = thold2 = (total_duration - tdec - vdep) / 2
            tdep = vdep
            # Ensure the calculated holding durations are valid.
            if thold1 < 0 or thold2 < 0:
                raise ValueError("Invalid protocol: Holding durations are negative.")

        # Append the results to the list.
        results.append({
            'Frequency (Hz)': freq,
            'Total Duration (ms)': total_duration,
            'Depolarizing Duration (tdep) (ms)': tdep,
            '1st Holding Duration (thold1) (ms)': thold1 if beat_type == 'square' else thold1,
            '2nd Holding Duration (thold2) (ms)': thold2 if beat_type == 'square' else None
        })

    # Convert the list of results to a DataFrame.
    results_df = pd.DataFrame(results)

    return results_df

def triangular_pacing(t, freq, incub = None, ca_i = None):
    
    """
    Triangular pacing function.

    Parameters:
    ----------
    t : float
        Time in minutes.
        
    freq : list
        List of frequencies (in Hz).
    
    vhold : int
        Holding potential (in mV).
    vdep : int
        Depolarizing voltage (in mV).
    tdec : int
        Decay duration (in ms) for triangular beat.
    steps : int
        Number of steps for the decay.
    beat_type : Boolean 
        Type of beat ('square' or 'triangular').

    Returns:
    ----------
    d_list : list
        A list with myokit datalogs.
    """
    
    # Initialize a list to store the results.
    d_list = list()
    peak_values = list()
    
    for i in freq:
        
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
        bcl = 1000/i
        
        # Add a v1 variable which defines the downward slope. 
        v1 = c.add_variable('v1')
        v1.set_rhs('40 - 1.2 * (engine.time % '+ str(bcl) +')')
        
        # Add a 'p' variable.
        vp = c.add_variable('vp')
        vp.set_rhs(0)
        vp.set_binding('pace')
        
        # Set a new right-hand side equation for V.
        v.set_rhs('piecewise((engine.time % ' + str(bcl) + ' >= 0 and engine.time % ' + str(bcl) + ' <= 100), v1, vp)')
        
        # Print the 'membrane' component.
        print(c.code())
        
        # Initialize a myokit protocol.
        p = myokit.Protocol()
        
        # Determine the amount of beats for the duration of simulation.
        num_beats = int(t * i * 60)
        
        # Loop through the amount of beats for each frequency. 
        for i in range(num_beats):
            # Add the holding potential steps to the protocol.
            p.add_step(-80, 100)
            p.add_step(-80, bcl - 100)
            
        # Define the total time characteristic.
        t_tot = p.characteristic_time() - 1
            
        # Compile the simulation.
        s = myokit.Simulation(m, p)
        
        # Set the maximum timestep size and tolerances
        s.set_max_step_size(0.1)
        s.set_tolerance(1e-8, 1e-8)
        
        # Set the incubation state as starting point.
        if incub is not None:
            s.set_state(incub)
            
        if ca_i is not None:
            s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
    
        # Run the simulation for the total time of the protocol.
        d = s.run(t_tot, log=['engine.time', 'membrane.V', 'Ca_Concentrations.fTrap', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S'],
                  log_interval = 5)
        d_list.append(d)
        
        # Save the max ftrap of every beat
        peak_vals, peak_idx = find_peaks_in_data(d['Ca_Concentrations.fTrap'])
        
        # Normalize the amount of beats to represent 10 minutes
        time_ftrap = np.linspace(0, 10, len(peak_vals))
        
        # Combine into one DataFrame
        ftrap_time = pd.DataFrame({'Time': time_ftrap, 'Peak Value': peak_vals})

        # Append to list.
        peak_values.append(ftrap_time)
        
    return dict(d_list = d_list, max_ftrap = peak_values)

def trian_square_pacing(t, freq, incub_state, ca_i = None, ca_block = False):
    
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
    
    # Set the constant.
    #s.set_constant('I_SK_trafficking.x', x)
    
    # Set the LTCC block if true.
    if ca_block is True:
        s.set_constant('I_Ca.ca_block', 0.3)
    
    # To limit automaticity under demote.
    if ca_i is not None:
        s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        
    # Set the incubated state.
    s.set_state(incub_state)

    # Run the simulation for the total time of the protocol.
    d = s.run(t_trian, log=['engine.time', 'membrane.V', 'Ca_Concentrations.fTrap', 'Ca_Concentrations.Ca_sl', 'Ca_Concentrations.Ca_j',
                            'SR_Ca_Concentrations.Ca_sr',
                            'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S'], log_interval = 5)
    
    # Obtain the state after the triangular pacing
    state = s.state()
    
    # Reload the model again.
    m2 = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # Get membrane potential and level
    v2 = m2.get('membrane.V')
    l2 = m2.get('membrane.level')

    # Unbind the membrane potential.
    l2.set_binding(None)
    
    # Demote the voltage variable and bind to pace
    v2.demote()
    v2.set_rhs(0)
    v2.set_binding('pace')
    
    if ca_i is not None:
        # Set the intracellular Ca2+ concentration to 500 nM.
        cai = m2.get('Ca_Concentrations.Ca_i')
        # Demote cai.
        cai.demote()
        # Set rhs to 500 nM but units are mM.
        cai.set_rhs(ca_i)
    
    # Add a square voltage step protocol after triangular pacing based on Figure 7 from Heijman et al. 2023. Circ Res.
    volt_steps = np.arange(-121, 81, 10)
    p2 = myokit.Protocol()
    for k, step in enumerate(volt_steps):
        p2.add_step(-50, 2350) #
        p2.add_step(step, 300)
        p2.add_step(-50, 2350) 
    
    # Determine the characteristic time.
    t_square = p2.characteristic_time()

    # Set the protocol.
    s2 = myokit.Simulation(m2, p2)
    
    # Set the state for the protocol.
    s2.set_state(state)
    
    # Set the constant.
    #s2.set_constant('I_SK_trafficking.x', x)
    
    # Set the LTCC block if true.
    if ca_block is True:
        s2.set_constant('I_Ca.ca_block', 0.3)
        
    # To limit automaticity under demote
    if ca_i is not None:
        s2.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
            
    # Set the maximum timestep size and tolerances.
    s2.set_max_step_size(0.1)
    s2.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation.
    d2 = s2.run(t_square, log=['engine.time', 'membrane.V', 'Ca_Concentrations.fTrap', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S',
                               'Ca_Concentrations.Ca_sl', 'I_Ca.I_Ca', 'SR_Ca_Concentrations.Ca_sr'])
    
    # Split the log.
    ds = d2.split_periodic(5000, adjust = True)
    
    # Save the min or max from each pulse based on the step value.
    max_sk = np.zeros(len(ds))
    for k, dd in enumerate(ds):
        temp = dd.trim_left(100, adjust=True)
        ds[k] = temp
        step_value = volt_steps[k]
        if step_value < -80:
            max_sk[k] = min(temp['I_SK.I_sk'])
        else:
            max_sk[k] = max(temp['I_SK.I_sk'])
            
    final_state = s2.state()
       
    return dict(trian = d, square = d2, max_sk = max_sk, final_state = final_state, trian_state = state)

def figure_1C(incub_state, ca_i = None):

    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
        
    # Get membrane potential and level
    v = m.get('membrane.V')
    l = m.get('membrane.level')
    
    # Unbind the membrane potential.
    l.set_binding(None)
    
    # Demote the voltage variable and bind to pace
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace')
    
    if ca_i is not None:
        # Set the intracellular Ca2+ concentration to 500 nM.
        cai = m.get('Ca_Concentrations.Ca_i')
        # Demote nai 
        cai.demote()
        # Set rhs to 500 nM but units are mM.
        cai.set_rhs(ca_i)
    
    # Define the voltage steps.
    volt_steps = np.arange(-121, 81, 10)
    
    # Initiate a protocol and step through the voltage steps.
    p = myokit.Protocol()
    for k, step in enumerate(volt_steps):
        p.add_step(-50, 2350) 
        p.add_step(step, 300)
        p.add_step(-50, 2350) 
    
    # Determine the characteristic time.
    t = p.characteristic_time()
    
    # Set the protocol.
    s = myokit.Simulation(m, p)
    
    # Set the maximum timestep size and tolerances
    s.set_max_step_size(0.1)
    s.set_tolerance(1e-8, 1e-8)
    
    # Set the incub state.
    s.set_state(incub_state)
    
    # To limit automaticity under demote
    if ca_i is not None:
        s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        
    # Run the simulation.
    d = s.run(t, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'Ca_Concentrations.Ca_sl', 'Ca_Concentrations.Ca_j', 'I_SK_trafficking.M', 'I_SK_trafficking.S'])
    
    # Split the log.
    ds = d.split_periodic(5000, adjust = True)
    
    # Save the min or max from each pulse based on the step value.
    max_sk = np.zeros(len(ds))
    for k, dd in enumerate(ds):
        temp = dd.trim_left(100, adjust=True)
        ds[k] = temp
        step_value = volt_steps[k]
        if step_value < -80:
            max_sk[k] = min(temp['I_SK.I_sk'])
        else:
            max_sk[k] = max(temp['I_SK.I_sk'])

        
    return dict(d = d, max_sk = max_sk, incub_state = incub_state)

def dem_reg_pacing(freq, t, koca = 10, ex_ca = False, ca_i = None, ca_block = False):
    
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
    s.set_max_step_size(1)
    s.set_tolerance(1e-8, 1e-8)
    
    # Set the koCa parameter
    s.set_constant('parameters.koCaBase', koca)
    
    # Set the LTCC block if true.
    if ca_block is True:
        s.set_constant('I_Ca.ca_block', 0.25)
        #s.set_constant('I_NCX.block_ncx', 0)
    
    if ex_ca is True:
        s.set_constant('parameters.Cao', ex_ca)

    # Run the simulation for the total time of the protocol.
    d = s.run(t_trian, log=['engine.time', 'membrane.V', 'Ca_Concentrations.fTrap', 'Ca_Concentrations.Ca_sl', 
                            'Ca_Concentrations.Ca_j', 'SR_Ca_Concentrations.Ca_sr', 'I_Ca.I_Ca', 'SR_Fluxes.J_SRCarel'])
  
    return d

def plot_dem_reg(dem, reg, freq, ca_i, koca = None, ca_block = False, last_beats = False, low_cao = False):
    
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].plot(dem['engine.time'], dem['membrane.V'], 'blue', label = f"Demote (Ca: {ca_i} mM)")
    ax[0, 0].plot(reg['engine.time'], reg['membrane.V'], ls = 'dashed', label = 'Regular')
    ax[0, 0].set_xlabel('Time [ms]')
    ax[0, 0].set_ylabel('Membrane potential [mV]')
    ax[0, 0].set_title('Membrane potential')
    ax[0, 0].legend(loc = 'upper left')
    if last_beats:
        ax[0, 0].set_xlim([110000, 120000])
    

    ax[1, 0].plot(dem['engine.time'], dem['Ca_Concentrations.Ca_sl'], 'green', label = f"Demote (Ca: {ca_i} mM)")
    ax[1, 0].plot(reg['engine.time'], reg['Ca_Concentrations.Ca_sl'], ls = 'dashed', label = 'Regular')
    ax[1, 0].set_xlabel('Time [ms]')
    ax[1, 0].set_ylabel('[Ca_sl]')
    ax[1, 0].set_title('Ca2+ concentration sarcolemmal')
    ax[1, 0].legend(loc = 'upper left')
    if last_beats:
        ax[1, 0].set_xlim([110000, 120000])

    ax[2, 0].plot(dem['engine.time'], dem['Ca_Concentrations.Ca_j'], 'red', label = f"Demote (Ca: {ca_i} mM)")
    ax[2, 0].plot(reg['engine.time'], reg['Ca_Concentrations.Ca_j'], ls = 'dashed', label = 'Regular')
    ax[2, 0].set_xlabel('Time [ms]')
    ax[2, 0].set_ylabel('[Ca_j]')
    ax[2, 0].set_title('Ca2+ concentration junctional')
    ax[2, 0].legend(loc = 'upper left')
    if last_beats:
        ax[2, 0].set_xlim([110000, 120000])

    ax[0, 1].plot(dem['engine.time'], dem['SR_Ca_Concentrations.Ca_sr'], 'k', label = f"Demote (Ca: {ca_i} mM)")
    ax[0, 1].plot(reg['engine.time'], reg['SR_Ca_Concentrations.Ca_sr'], ls = 'dashed', label = 'Regular')
    ax[0, 1].set_xlabel('Time [ms]')
    ax[0, 1].set_ylabel('[SR_Ca]')
    ax[0, 1].set_title('SR Ca2+ concentration')
    ax[0, 1].legend(loc = 'upper left')
    if last_beats:
        ax[0, 1].set_xlim([110000, 120000])

    ax[1, 1].plot(dem['engine.time'], dem['I_Ca.I_Ca'], 'purple', label = f"Demote (Ca: {ca_i} mM)")
    ax[1, 1].plot(reg['engine.time'], reg['I_Ca.I_Ca'], ls = 'dashed', label = 'Regular')
    ax[1, 1].set_xlabel('Time [ms]')
    ax[1, 1].set_ylabel('ICa')
    ax[1, 1].set_title('Ca2+ current')
    ax[1, 1].legend(loc = 'upper left')
    if last_beats:
        ax[1, 1].set_xlim([110000, 120000])
    
    ax[2, 1].plot(dem['engine.time'], dem['Ca_Concentrations.fTrap'], 'purple', label = f"Demote (Ca: {ca_i} mM)")
    ax[2, 1].plot(reg['engine.time'], reg['Ca_Concentrations.fTrap'], ls = 'dashed', label = 'Regular')
    ax[2, 1].set_xlabel('Time [ms]')
    ax[2, 1].set_ylabel('ftrap')
    ax[2, 1].set_title('ftrap')
    ax[2, 1].legend(loc = 'upper left')
    # if last_beats:
    #     ax[2, 1].set_xlim([100000, 120000])
    
  # Setting the suptitle based on the conditions
    suptitle = f'Frequency = {freq} Hz'
    if ca_block:
        suptitle += ' w/ LTCC block'
    if low_cao:
        suptitle += ' and Cao 0.2 mM'
    if koca is not None:
        suptitle += f' and koCa {koca}'
        
    plt.suptitle(suptitle)


    plt.tight_layout()
    plt.show()
    
def max_ftrap(x):
    
    # Initialize a list.
    max_ftrap = list()
    
    # Loop through the list and store the results.
    for i in range(len(x)):
        max_ftrap.append(max(x[i]['Ca_Concentrations.fTrap']))
        
    return max_ftrap


def plot_koca(frequency_labels, dem_values, title):
    # Calculate the steady state fTrap for each frequency
    ftrap_values = [max_ftrap(dem) for dem in dem_values]
    
    # Create the DataFrame
    df_koca = pd.DataFrame({'Frequency': frequency_labels,
                            'fTrap': ftrap_values})

    # Split the lists in 'fTrap' into separate columns
    df_koca_expanded = pd.DataFrame(df_koca['fTrap'].to_list(), columns=['Dem.', 'Reg.', 'Dem. Block', 'Reg. Block'])
    df_koca_expanded['Frequency'] = df_koca['Frequency']

    # Plot the data for each condition
    plt.figure()
    for condition in ['Dem.', 'Reg.', 'Dem. Block', 'Reg. Block']:
        plt.plot(df_koca_expanded['Frequency'], df_koca_expanded[condition], marker='o', linestyle='-', label=condition)

    plt.xlabel('Frequency')
    plt.ylabel('fTrap')
    plt.title(title)
    plt.legend(title='Conditions')
    plt.grid(True)
    plt.show()
    
def incub_prot(pp, ca_i = None, tblock = False):
    
    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # Obtain the parameters of interest.
    cai_idx = m.get('Ca_Concentrations.Ca_i').index()
    v_idx = m.get('membrane.V').index()
    memv_idx = v_idx + 1
    
    # Demote intracellular Ca2+ if needed. 
    if ca_i is not None:
        # Set the intracellular Ca2+ concentration to 500 nM.
        cai = m.get('Ca_Concentrations.Ca_i')
        # Demote cai.
        cai.demote()
        # Set rhs to 500 nM but units are mM.
        cai.set_rhs(ca_i)

    # Create an incubation protocol; so no stimulus.
    p_incub = myokit.Protocol()
    p_incub.schedule(0, 0, 1000, 0)
    
    # Create the simulation.
    s_incub = myokit.Simulation(m, p_incub)
    
    # To limit automaticity under demote
    if ca_i is not None:
        s_incub.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
    
    if tblock is True:
        s_incub.set_constant('I_SK_trafficking.x', 0)
    
    # Prepace the model.
    s_incub.pre(pp)
    
    # Return the states
    if ca_i is None:    
        incub_state_comp = s_incub.state()
        incub_state_cai = incub_state_comp.copy()
        del incub_state_cai[cai_idx]
        incub_state_v = incub_state_comp[memv_idx: ]
        incub_state_vcai = incub_state_comp[memv_idx: ] 
        del incub_state_vcai[cai_idx]
        return dict(incub_comp = incub_state_comp, incub_cai = incub_state_cai, incub_v = incub_state_v, incub_vcai = incub_state_vcai)
    
    else:
        incub_state_cai = s_incub.state()
        incub_state_vcai = incub_state_cai[memv_idx: ]
        return dict(incub_cai = incub_state_cai, incub_vcai = incub_state_vcai)
        

def current_prepace(pp, freq, ca_i = None, t_block = False):
   
    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # Obtain the indexes of the parameters of interest.
    cai_idx = m.get('Ca_Concentrations.Ca_i').index()
    v_idx = m.get('membrane.V').index()
    memv_idx = v_idx + 1
    
    # Demote intracellular Ca2+ if needed.
    if ca_i is not None:
        # Set the intracellular Ca2+ concentration to 500 nM.
        cai = m.get('Ca_Concentrations.Ca_i')
        # Demote cai.
        cai.demote()
        # Set rhs to 500 nM but units are mM.
        cai.set_rhs(ca_i)

    # Calculate the bcl.
    bcl = 1000/freq
    
    # Create a protocol.
    p = myokit.Protocol()
    p.schedule(1, 20, 5, bcl, 0) 
    
    # Create the simulation and pre-pace the model.
    s = myokit.Simulation(m, p)
    if ca_i is not None:
        s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
    
    if t_block is True:
        s.set_constant('I_SK_trafficking.x', 0)
        
    # Pre-pace the model.
    s.pre(bcl * pp)
    
    # Return the states.
    if ca_i is None:    
        incub_state_comp = s.state()
        incub_state_cai = incub_state_comp.copy()
        del incub_state_cai[cai_idx]
        incub_state_v = incub_state_comp[memv_idx: ]
        incub_state_vcai = incub_state_comp[memv_idx: ] 
        del incub_state_vcai[cai_idx]
        return dict(incub_comp = incub_state_comp, incub_cai = incub_state_cai, incub_v = incub_state_v, incub_vcai = incub_state_vcai)
    
    else:
        incub_state_cai = s.state()
        incub_state_vcai = incub_state_cai[memv_idx: ]
        return dict(incub_cai = incub_state_cai, incub_vcai = incub_state_vcai)
    
    
def find_peaks_in_data(data, height=None, distance=None, prominence=None):
    # Convert data to a numpy array if it's not already
    data_array = np.array(data)
    
    # Find peaks in the data
    peaks, _ = find_peaks(data_array, height=height, distance=distance, prominence=prominence)
    
    # Extract the peak values
    peak_values = data_array[peaks]
    
    return list(peak_values), peaks

def ftrap_freq_export(dictionary, data, freq, exp_name):
    
    # Iterate over each list and save it to a CSV file.
    for data_list, filename in zip(dictionary[data], freq):
        
        # Generate the filename dynamically incorporating the column names
        export_filename = f'Results/{exp_name}_{filename}.csv'
        
        # Save the combined DataFrame to a CSV file
        data_list.to_csv(export_filename, index=False)
        
        print(f"Data saved to {export_filename}")
        
    
def peak_filter(data, showit=False):
    """
    Identify and filter out invalid peaks (based on a membrane potential threshold) from membrane potential data.

    Parameters:
    ----------
    data : dict
        Dictionary containing the membrane potential data, where `data['d']['membrane.V']` holds the membrane potential values.
    showit : bool, optional
        If True, a plot of the membrane potential with all peaks and filtered valid peaks will be displayed. Default is True.

    Returns:
    ----------
    invalid_peak_counts : list
        A list of indices (corresponding to the peak counts) where the membrane potential values are less than or equal to -20 mV.
    """
    
    # Load the membrane potential.
    vm = np.array(data['d']['membrane.V'])
    
    # Step 1: Identify all peaks in the membrane potential data
    peaks, _ = find_peaks(vm)
    
    # Step 2: Identify the peak counts that are invalid
    invalid_peak_counts = [i for i, peak_idx in enumerate(peaks) if vm[peak_idx] <= 0]  # Invalid peak counts
    valid_peak_counts = [i for i, peak_idx in enumerate(peaks) if vm[peak_idx] > 0]     # Valid peak counts

    # Step 3: Remove the invalid peaks from the peaks array
    peaks_filtered = np.delete(peaks, invalid_peak_counts)
    
    if showit:
        plt.plot(vm, label='Membrane Potential')
        plt.scatter(peaks, vm[peaks], color='red', marker='o', label='All Peaks')
        plt.scatter(peaks_filtered, vm[peaks_filtered], color='green', marker='x', label='Valid Peaks (> 0 mV)')
        plt.xlabel('Time (datapoints)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Membrane Potential with Peaks')
        plt.legend()
        plt.show()
    
    # Step 5: Return the peak count indices of the invalid peaks
    if len(invalid_peak_counts) > 0:
        print("Invalid peaks (<= 0 mV) detected at peak counts:", invalid_peak_counts)
    else:
        print("No invalid peaks detected.")
        
    return invalid_peak_counts


def calc_apd90(data, freq, t, cutoff):
    """
    Calculate and adjust APD90 values for a given pacing frequency and duration, inserting NaNs for missing beats
    and filtering out invalid peaks.

    Parameters:
    ----------
    data : dict
        Dictionary containing the APD90 data, where `data['apd']['duration']` holds the APD90 values.
    freq : float
        Pacing frequency in Hz.
    t : int
        Duration of the pacing period in minutes.
    cutoff : float
        Cutoff value for APD90. Only values below this cutoff will be considered valid.

    Returns:
    ----------
    results : dict
        A dictionary containing:
        - 'time': numpy array representing the time axis in minutes.
        - 'apd90': numpy array with the adjusted APD90 values, including NaNs for missing beats and only valid data points.
        - 'first': The first valid APD90 value that met the cutoff criterion.
    """

    # Load the APD90 data
    apd90 = np.array(data['apd']['duration'])

    # Define pacing rate and total duration in minutes
    pacing_rate_hz = freq  # Pacing rate in Hz
    duration_minutes = t  # Total duration in minutes
    beats_per_minute = pacing_rate_hz * 60  # Number of beats per minute
    expected_beats = int(beats_per_minute * duration_minutes)  # Total expected number of beats

    # If APD90 has more data points than expected, filter out invalid peaks
    if len(apd90) > expected_beats:
        invalid_peak_counts = peak_filter(data=data, showit=False)
        apd90 = np.delete(apd90, invalid_peak_counts)

    # Initialize a new array of NaNs with the length of expected beats
    new_apd90 = np.full(expected_beats, np.nan)

    # Find the first valid beat where APD90 < cutoff
    if cutoff is not None:
        first_valid_index = np.argmax(apd90 < cutoff)  # Get the index of the first valid beat
    else:
        first_valid_index = 0

    # Insert valid APD90 values after the first valid beat's position
    num_valid_beats = len(apd90[first_valid_index:])  # Calculate how many valid beats there are
    new_apd90[-num_valid_beats:] = apd90[first_valid_index:]  # Insert valid beats at the end of the array

    # Enforce the rule: replace values greater than the cutoff with NaN after the first valid index
    found_cutoff = False
    for i in range(first_valid_index, len(new_apd90)):
        if new_apd90[i] < cutoff:
            found_cutoff = True
        elif found_cutoff and new_apd90[i] > cutoff:
            new_apd90[i] = np.nan

    # Find the first valid measurement value for reference
    first_real_value = apd90[first_valid_index] if first_valid_index < len(apd90) else np.nan

    # Generate the time axis in minutes
    time_axis = np.linspace(0, duration_minutes, expected_beats)
    
    return dict(time=time_axis, apd90=new_apd90, first=first_real_value)


def calculate_reentry_time(i, vm, dur_sim, dur, s2, interval, cutoff):
    """
    Calculates the reentry time based on the given parameters.

    Parameters:
        i (int): Reentry iteration.
            The iteration count of the reentry process.
        vm (numpy.ndarray): Membrane potential.
            The membrane potential data.
        dur_sim (float): Duration of the S1S2 simulation.
            The total duration of the S1S2 simulation.
        dur (float): Duration of the S1 stimulus.
            The duration of the S1 stimulus.
        s2 (float): Time point for the S2 stimulus.
            The time point at which the S2 stimulus is applied.
        interval (float): Recording interval.
            The time interval between recordings.
        cutoff (float): Membrane potential regarded as inactivity.
            The threshold value below which the membrane potential is considered inactive.

    Returns:
        int: Reentry time in milliseconds.
            The duration of reentry, measured in milliseconds.
    """
    # Calculate the total duration of the reentry simulation in data points.
    iteration = i + 1
    tot_dur = ((dur_sim * iteration) / interval) + (dur / interval) - 1

    # Determine the index corresponding to the final S2 stimulus.
    final_stim = int(s2 / interval + 1)

    # Extract the membrane potential data after the final S2 stimulus.
    remain_vm = vm[final_stim:, :, :]

    # Find the maximum membrane potential after the final S2 stimulus for each recording.
    maxval = np.max(remain_vm, axis=(1, 2))

    # Find the indices where the maximum membrane potential falls below the cutoff value.
    reentry_stop_indices = np.nonzero(maxval < cutoff)[0]

    if i == 0:
        # For the first iteration, determine the reentry time based on the first occurrence of inactivity.
        time_stop = int((final_stim + reentry_stop_indices[0]) * interval)
    else:
        # For subsequent iterations, calculate the reentry time based on the last stimulus and reentry stop index.
        final_stim = int(tot_dur - (dur_sim / interval))

        if reentry_stop_indices.size > 0:
            reentry_stop = reentry_stop_indices[0]
            t = (final_stim + reentry_stop) * interval
        else:
            # If no reentry stop is found, the reentry continues until the end of the simulation.
            reentry_stop = None
            t = tot_dur * interval

        time_stop = int(t)

    # Calculate the reentry time in milliseconds.
    reentry_time_val = time_stop - s2
    
    # Print the reentry time.
    print(f'The reentry time was {reentry_time_val} ms')
    
    return reentry_time_val

def voltage_clamp_prot(protocol_type="triangular", plot=True):
    """
    Generates voltage clamp protocol data for either a triangular or square pulse protocol.

    Parameters:
    ----------
    protocol_type : str, optional
        Type of protocol to generate. Options are "triangular" or "square". Default is "triangular".
    
    plot : bool, optional
        If True, plots the generated protocol. Default is True.
    
    Returns:
    ----------
    data : pd.DataFrame
        DataFrame containing the time and voltage ('voltage') data from the protocol.
    """
    
    if protocol_type == "triangular":
        # Parameters for triangular protocol
        holding_potential = -80   # mV, the baseline holding potential before depolarization
        peak_potential = 40       # mV, the peak depolarization voltage
        ramp_duration = 100       # ms, the duration for the depolarization ramp
        interval_duration = 200   # ms, the interval between two cycles
        step_duration = 1         # ms, the duration for instantaneous steps in the protocol
        holding_period = 100      # ms, the period spent at the holding potential before ramp starts
        total_time = holding_period + 2 * interval_duration  # Total time for one cycle

        # Define time points with sufficient resolution for smooth plotting (1000 points)
        time = np.linspace(0, total_time, 100)  
        voltage = np.full_like(time, holding_potential)  # Initialize voltage to holding potential

        # Define the start and end times for each segment of the triangular protocol
        cycle1_start = holding_period
        cycle1_ramp_end = cycle1_start + step_duration + ramp_duration
        cycle2_start = cycle1_start + interval_duration
        cycle2_ramp_end = cycle2_start + step_duration + ramp_duration

        # First depolarization cycle: Instantaneous step to peak, followed by ramp down to holding potential
        voltage[time >= cycle1_start] = holding_potential  # Keep the voltage at holding potential initially
        voltage[(time >= cycle1_start) & (time < cycle1_start + step_duration)] = peak_potential  # Instantaneous step to peak potential
        ramp_time1 = time[(time >= cycle1_start + step_duration) & (time < cycle1_ramp_end)]  # Time for the ramp
        voltage[(time >= cycle1_start + step_duration) & (time < cycle1_ramp_end)] = np.linspace(peak_potential, holding_potential, ramp_time1.size)  # Linearly interpolate voltage during ramp

        # Second depolarization cycle: Same as the first, but after an interval period
        voltage[(time >= cycle2_start) & (time < cycle2_start + step_duration)] = peak_potential  # Instantaneous step to peak potential
        ramp_time2 = time[(time >= cycle2_start + step_duration) & (time < cycle2_ramp_end)]  # Time for the second ramp
        voltage[(time >= cycle2_start + step_duration) & (time < cycle2_ramp_end)] = np.linspace(peak_potential, holding_potential, ramp_time2.size)  # Linearly interpolate voltage during ramp
        
    elif protocol_type == "square":
        # Voltage steps (from -120 to 80 in steps of 10)
        voltage_steps = np.arange(-120, 90, 10)  # Define voltage steps to apply during the square pulse

        # Define the protocol durations
        holding_time = 100  # ms (time at -80 mV before applying the voltage step)
        step_time = 300     # ms (duration of each voltage step)

        # Initialize lists to store time and voltage data
        time_list = []
        voltage_list = []

        # Loop over each voltage step and create the protocol
        for voltage_step in voltage_steps:
            # 100 ms at -80 mV (holding voltage before the step)
            time_list.append(np.linspace(0, holding_time, holding_time, endpoint=False))  # Time for holding
            voltage_list.append(np.full(holding_time, -50))  # Voltage for holding

            # 300 ms at the current voltage step
            time_list.append(np.linspace(holding_time, holding_time + step_time, step_time, endpoint=False))  # Time for voltage step
            voltage_list.append(np.full(step_time, voltage_step))  # Voltage for the current step

            # 100 ms back to -80 mV after each step
            time_list.append(np.linspace(holding_time + step_time, holding_time + step_time + holding_time, holding_time, endpoint=False))  # Time for returning to -80
            voltage_list.append(np.full(holding_time, -50))  # Voltage for returning to 50

        # Concatenate the time and voltage data from all steps into single arrays
        time = np.concatenate(time_list)
        voltage = np.concatenate(voltage_list)
    
    else:
        # Raise an error if an invalid protocol type is provided
        raise ValueError("Invalid protocol_type. Choose either 'triangular' or 'square'.")

    # Create a DataFrame with time and voltage columns
    data = pd.DataFrame({
        "time": time,
        "voltage": voltage
    })

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 5))  # Set up the figure for plotting
        
        if protocol_type == "triangular":
            plt.plot(data["time"], data["voltage"], label="Triangular Pulse")
        
        elif protocol_type == "square":
            plt.plot(data["time"], data["voltage"], label="Square Pulse")
       
        # Set labels and title for the plot
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"{protocol_type.capitalize()} Voltage Clamp Protocol")
        plt.ylim(-130, 90)  # Set the y-axis limits for better visualization
        plt.legend(loc="upper right")  # Display legend in the upper right corner
        plt.grid(True)  # Add grid lines for better readability
        plt.show()  # Show the plot
    
    return data  # Return the DataFrame with the time and voltage data

def reversibility(freq, t, incub, interval, ca_i=None, current_clamp=False, t_block=False, ca_block=False):
    """
    Simulates reversibility using either current clamp or voltage clamp mode and optionally blocking certain
    ionic currents and/or setting intracellular calcium concentration.

    Parameters:
    ----------
    freq : float
        Stimulation frequency in Hz.
    t : int
        Duration of the pacing protocol in minutes.
    incub : list
        Initial incubated state for the model.
    interval : float
        Interval for logging the data in milliseconds.
    ca_i : float, optional
        Intracellular calcium concentration to use if Ca2+ buffering is to be applied. Default is None.
    current_clamp : bool, optional
        If True, runs the simulation in current clamp mode. Default is False (voltage clamp mode).
    t_block : bool, optional
        If True, applies trafficking block. Default is False.
    ca_block : bool, optional
        If True, applies Ca2+ channel block. Default is False.

    Returns:
    ----------
    dict
        Contains the following keys:
        - 'd': Simulation data log (time course data).
        - 'state': Final state of the model after simulation.
        - 'apd': Action potential duration data (only in current clamp mode).
        - 'vt': Voltage threshold for APD calculation (only in current clamp mode).
    """
    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # If intracellular calcium buffering is specified, set it to the desired concentration
    if ca_i is not None:
        # Access the intracellular Ca2+ concentration component in the model
        cai = m.get('Ca_Concentrations.Ca_i')
        cai.demote()  # Remove default equation for intracellular calcium
        cai.set_rhs(ca_i)  # Set concentration to specified ca_i in mM
    
    # Calculate the basic cycle length (BCL) in ms from the frequency in Hz
    bcl = 1000 / freq
    
    # Branch between current clamp and voltage clamp modes
    if current_clamp:
        # In current clamp mode, create a pacing protocol and set the clamp to run simulation
        p = myokit.Protocol()
        p.schedule(1, 20, 5, bcl, 0)  # Defines pacing protocol with pulse at 20 ms and period equal to BCL

        # Compile the simulation with the pacing protocol
        s = myokit.Simulation(m, p)
        
        # Set simulation tolerances and step sizes
        s.set_max_step_size(0.1)
        s.set_tolerance(1e-8, 1e-8)
        
        # If calcium buffering is used, limit automaticity by reducing sarcoplasmic reticulum calcium release rate
        if ca_i is not None:
            s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        
        # Apply calcium channel blocking if specified
        if ca_block:
            s.set_constant('I_Ca.ca_block', 0.3)
        
        # Apply trafficking block if specified
        if t_block:
            s.set_constant('I_SK_trafficking.x', 0)
        
        # Set the initial state to the incubated state provided as input
        s.set_state(incub)
        
        # Calculate total simulation time in ms and the number of beats to simulate
        time_ms = t * 60000
        num_beats = time_ms / bcl
        
        # Set voltage threshold for APD90 calculation (90% of initial membrane potential)
        vt = 0.9 * s.state()[m.get('membrane.V').index()]
        
        # Run simulation and record data with a specified logging interval
        if ca_i is not None:
            d, apd = s.run(bcl * num_beats, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M',
                                                 'I_SK_trafficking.S', 'Ca_Concentrations.fTrap', 
                                                 'Ca_Concentrations.Ca_sl', 'SR_Ca_Concentrations.Ca_sr',
                                                 'Na_Concentrations.Na_i', 'I_NCX.I_ncx'],
                           apd_variable='membrane.V', apd_threshold=vt, log_interval=interval)
        else:
            d, apd = s.run(bcl * num_beats, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M',
                                                 'I_SK_trafficking.S', 'Ca_Concentrations.fTrap', 
                                                 'Ca_Concentrations.Ca_sl', 'SR_Ca_Concentrations.Ca_sr',
                                                 'Ca_Concentrations.Ca_i', 'Na_Concentrations.Na_i', 
                                                 'I_NCX.I_ncx'],
                           apd_variable='membrane.V', apd_threshold=vt, log_interval=interval)
        
        # Get the final state of the simulation
        state = s.state()
        
        # Output a statement about the simulation conditions
        if ca_i is not None:
            print(f'Current clamp performed at {freq} Hz for {t} minutes (Ca2+ buffered = {ca_i} mM)')
        else:
            print(f'Current clamp performed at {freq} Hz for {t} minutes (Free Ca2+)')

        # Return a dictionary with the simulation results
        return dict(d=d, state=state, apd=apd, vt=vt)

    else:
        # In voltage clamp mode, configure the model for voltage protocol
        c = m.get('membrane')
        v = c.get('V')
        v.set_binding(None)
        v.demote()  # Unbind and demote V variable
        
        l = c.get('level')  # Access membrane level variable and unbind it
        l.set_binding(None)
        
        # Create new variables to define pacing and voltage clamp protocol
        v1 = c.add_variable('v1')
        v1.set_rhs(f'40 - 1.2 * (engine.time % {bcl})')  # Ramp protocol equation
        
        vp = c.add_variable('vp')
        vp.set_rhs(0)
        vp.set_binding('pace')  # Bind vp variable to pacing

        # Set the piecewise voltage protocol for membrane potential
        v.set_rhs(f'piecewise((engine.time % {bcl} >= 0 and engine.time % {bcl} <= 100), v1, vp)')
        
        # Initialize protocol to control voltage clamp
        p = myokit.Protocol()
        
        # Determine number of beats to simulate based on frequency and duration
        num_beats = int(t * freq * 60)
        
        # Define protocol steps for each beat
        for i in range(num_beats):
            p.add_step(-80, 100)  # Holding potential
            p.add_step(-80, bcl - 100)  # Interval between beats
        
        # Define the simulation time based on protocol length
        time = p.characteristic_time() - 1
            
        # Compile the simulation with the protocol
        s = myokit.Simulation(m, p)
        
        # Set simulation tolerances and step sizes
        s.set_max_step_size(0.1)
        s.set_tolerance(1e-8, 1e-8)
        
        # Set the initial state to the incubated state provided as input
        s.set_state(incub)
        
        # If calcium buffering is used, limit automaticity by reducing sarcoplasmic reticulum calcium release rate
        if ca_i is not None:
            s.set_constant('SR_Fluxes.koSRCa_freq', 0.25)
        
        # Apply calcium channel block if specified
        if ca_block:
            s.set_constant('I_Ca.ca_block', 0.3)
        
        # Apply trafficking block if specified
        if t_block:
            s.set_constant('I_SK_trafficking.x', 0)
        
        # Run simulation and record data with specified logging interval
        if ca_i is not None:
            d = s.run(time, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S',
                                 'Ca_Concentrations.fTrap', 'Ca_Concentrations.Ca_sl', 'SR_Ca_Concentrations.Ca_sr',
                                 'Na_Concentrations.Na_i', 'I_NCX.I_ncx'],
                      log_interval=interval)
        else:
            d = s.run(time, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S',
                                 'Ca_Concentrations.fTrap', 'Ca_Concentrations.Ca_sl', 'SR_Ca_Concentrations.Ca_sr',
                                 'Ca_Concentrations.Ca_i', 'Na_Concentrations.Na_i', 'I_NCX.I_ncx'],
                      log_interval=interval)
        
        # Get the final state of the simulation
        state = s.state()
        
        # Output a statement about the simulation conditions
        if ca_i is not None:
            print(f'Voltage clamp performed at {freq} Hz for {t} minutes (Ca2+ buffered = {ca_i} mM).')
        else:
            print(f'Voltage clamp performed at {freq} Hz for {t} minutes (Free Ca2+).')
    
        # Return a dictionary with the simulation results
        return dict(d=d, state=state)

 
def export_PoM_results_by_repeat(PoM_res, base_name='PoM'):
    results = PoM_res['results']
    apds = PoM_res['apd']
    param_vals = PoM_res['param_vals']
    n_repeats = len(results)
    n_sims = len(results[0])

    # Define variables and filenames (base)
    variables = ['membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M']
    variable_labels = ['V', 'ISK', 'trafficking_M']

    for var_key, label in zip(variables, variable_labels):
        for j in range(n_repeats):
            all_sims = []
            time_vector = results[j][0]['engine.time']
            for i in range(n_sims):
                all_sims.append(results[j][i][var_key])
            df = pd.DataFrame(np.column_stack(all_sims),
                              columns=[f'sim{i+1}' for i in range(n_sims)])
            df.insert(0, 'time', time_vector)
            df.to_csv(f'Results/{base_name}_{label}_repeat{j+1}.csv', index=False)

    # Export APDs by repeat
    for j in range(n_repeats):
        df_apd = pd.DataFrame({
            'sim': [f'sim{i+1}' for i in range(n_sims)],
            'APD': apds[j]
        })
        df_apd.to_csv(f'Results/{base_name}_APDs_repeat{j+1}.csv', index=False)

    # Export parameter values by repeat
    param_names = PoM_res.get('param_names', [f'param_{i+1}' for i in range(len(param_vals[0][0]))])
    for j in range(n_repeats):
        df_params = pd.DataFrame(param_vals[j], columns=param_names)
        df_params.insert(0, 'sim', [f'sim{i+1}' for i in range(n_sims)])
        df_params.to_csv(f'Results/{base_name}_params_repeat{j+1}.csv', index=False)

    print("✅ Export complete. Files saved to 'Results/' (grouped by repeat).")
    
def PoM(m, freq1, freq2, t, incub, interval, param_names, param_vals, n_sims, n_repeats=1, seed=42):
    """
    Simulates a Population of Models (PoM) over two pacing protocols, each with a different frequency,
    with multiple stochastic parameter perturbations and repeats.

    Parameters:
    ----------
    m : myokit.Model
        The Myokit model object to simulate.

    freq1 : float
        Pacing frequency (in Hz) for Phase 1.

    freq2 : float
        Pacing frequency (in Hz) for Phase 2.

    t : float
        Duration (in minutes) of each pacing phase.

    incub : list
        The pre-paced model state to initialize the simulation with.

    interval : float
        Logging interval for time-series data (in ms).

    param_names : list of str
        List of parameter names to perturb in the simulations.

    param_vals : list of float
        Baseline values for the parameters listed in `param_names`.

    n_sims : int
        Number of simulations to run per repeat.

    n_repeats : int, optional
        Number of repeats of the full simulation set (default is 1).

    seed : int, optional
        Base random seed for reproducibility (default is 42).

    Returns:
    ----------
    dict
        A dictionary containing:
        - 'results': Time-series results for all simulations and repeats.
        - 'apd': Action potential durations (APDs) for each simulation.
        - 'param_vals': Perturbed parameter sets used in each simulation.
        - 'd1': Log from last simulation in Phase 1 (for inspection).
        - 'd2': Log from last simulation in Phase 2 (for inspection).
    """
    
    # Prepare data containers for simulation output
    combined_results = []
    combined_apd = []
    sampled_param_vals = []

    # Calculate basic cycle lengths (ms) from pacing frequencies
    bcl1 = 1000 / freq1
    bcl2 = 1000 / freq2

    # Define pacing protocols for each phase
    p1 = myokit.Protocol()
    p1.schedule(1, 20, 5, bcl1, 0)  # 1 ms stimuli of 5 ms duration at bcl1 intervals

    p2 = myokit.Protocol()
    p2.schedule(1, 20, 5, bcl2, 0)  # 1 ms stimuli of 5 ms duration at bcl2 intervals

    # Loop over repeat batches
    for j in range(n_repeats):
        repeat_results = []
        repeat_apds = []
        repeat_params = []

        # Loop over simulations per repeat
        for i in range(n_sims):
            rng = np.random.default_rng(seed + j * n_sims + i)  # Unique RNG per simulation

            # Generate perturbed parameter set: multiplicative N(1.0, 0.1)
            scalars = rng.normal(loc=1.0, scale=0.1, size=len(param_vals))
            sampled_vals = [s * p for s, p in zip(scalars, param_vals)]
            repeat_params.append(sampled_vals)

            # ------------------- Phase 1 Simulation -------------------
            s1 = myokit.Simulation(m, p1)
            for name, val in zip(param_names, sampled_vals):
                s1.set_constant(name, val)  # Apply perturbations
            s1.set_max_step_size(0.1)
            s1.set_tolerance(1e-8, 1e-8)
            s1.set_state(incub)  # Load incubation/pre-paced state

            num_beats1 = (t * 60000) / bcl1  # Convert minutes to ms and divide by BCL
            vt1 = 0.9 * s1.state()[m.get('membrane.V').index()]  # APD90 threshold (90% repol)

            # Run phase 1 and log key variables + APDs
            d1, apd1 = s1.run(bcl1 * num_beats1, 
                              log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M'],
                              apd_variable='membrane.V', apd_threshold=vt1, log_interval=interval)
            state_after_1 = s1.state()  # Use final state as input to phase 2

            print(f'[Repeat {j+1} | Sim {i+1}] Phase 1 done at {freq1} Hz')

            # ------------------- Phase 2 Simulation -------------------
            s2 = myokit.Simulation(m, p2)
            s2.set_max_step_size(0.1)
            s2.set_tolerance(1e-8, 1e-8)
            s2.set_state(state_after_1)

            num_beats2 = (t * 60000) / bcl2
            vt2 = 0.9 * s2.state()[m.get('membrane.V').index()]

            # Run phase 2 and log key variables + APDs
            d2, apd2 = s2.run(bcl2 * num_beats2,
                              log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M'],
                              apd_variable='membrane.V', apd_threshold=vt2, log_interval=interval)

            print(f'[Repeat {j+1} | Sim {i+1}] Phase 2 done at {freq2} Hz')

            # ------------------- Combine Logs -------------------
            # Adjust time axis of second phase for continuity
            t1 = np.asarray(d1['engine.time'])
            t2 = np.asarray(d2['engine.time']) + t1[-1]

            d_comb = {'engine.time': np.concatenate([t1, t2])}
            for key in d1:
                if key == 'engine.time':
                    continue
                d_comb[key] = np.concatenate([np.asarray(d1[key]), np.asarray(d2[key])])

            # Store simulation data
            repeat_results.append(d_comb)
            repeat_apds.append(apd1['duration'] + apd2['duration'])

        # Store repeat-level data
        combined_results.append(repeat_results)
        combined_apd.append(repeat_apds)
        sampled_param_vals.append(repeat_params)

    # Return full dataset
    return dict(results=combined_results, apd=combined_apd, param_vals=sampled_param_vals, d1=d1, d2=d2)

def correct_time_glitch(file_path, interval=5):
    """
    Corrects a time discontinuity in a CSV file caused by duplicated time entries
    (specifically at 599995.0 ms), typically due to improper log concatenation.

    Parameters:
    ----------
    file_path : str
        Path to the CSV file containing the time-series data to be corrected.

    interval : int, optional
        Time step interval (in ms) used to regenerate the corrected time series.
        Default is 5 ms.

    Returns:
    ----------
    None
        The function overwrites the original file with corrected time values.
        Prints status updates to indicate success or if correction was not needed.
    """
    
    # Read the time-series CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Target duplicated time value indicating where the glitch occurs
    dup_time = 599995.0

    # Find all indices where the duplicated time occurs
    dup_indices = df.index[df['time'] == dup_time].tolist()

    # If there's fewer than 2 duplicates, no correction needed
    if len(dup_indices) < 2:
        print(f"No duplicate time found at {dup_time} in {file_path}. Skipping.")
        return

    # Identify the second occurrence of the duplicate time
    second_index = dup_indices[1]

    # Determine starting time for corrected segment
    correction_start = df.loc[second_index - 1, 'time'] + interval

    # Generate corrected time values starting from the duplicate onward
    corrected_times = [correction_start + interval * i for i in range(len(df) - second_index)]

    # Apply the corrected time values back to the DataFrame
    df.loc[second_index:, 'time'] = corrected_times

    # Overwrite the file with corrected data
    df.to_csv(file_path, index=False)
    print(f"✅ Fixed time in {file_path} from index {second_index} onward.")

def PoM_sens_PLS(data, param_vals, param_names, interval=None, t=None, n_components=None, mode='channel'):
    """
    Performs Partial Least Squares (PLS) sensitivity analysis on either:
    - 'channel': Time-series data of membrane channel expression at a specified time point.
    - 'apd90': Scalar output data such as APD90 across simulations.
    
    Parameters:
    ----------
    data : pd.DataFrame or 1D array-like
        - If mode='channel': A DataFrame with time-resolved membrane channel outputs.
        - If mode='apd90': A 1D array-like of APD90 values per simulation.
    
    param_vals : pd.DataFrame
        DataFrame of perturbed parameter values, with 'sim' column included.
    
    param_names : list of str
        List of parameter names corresponding to columns in `param_vals`.
    
    interval : int, optional
        Logging interval in ms (only used in 'channel' mode).
    
    t : float, optional
        Time in minutes for data extraction (only used in 'channel' mode).
    
    n_components : int, optional
        Number of PLS components. Defaults to min(n_samples, n_features, 10).
    
    mode : str, {'channel', 'apd90'}
        Selects between membrane channel sensitivity or APD90 sensitivity.
    
    Returns:
    ----------
    pd.DataFrame
        DataFrame with sensitivity values for each input parameter.
    """
    
    # Prepare input matrix X: drop sim column, apply log-transform
    X = np.log(param_vals.drop(columns=['sim']).values)

    if mode == 'channel':
        if t is None or interval is None:
            raise ValueError("Both 't' and 'interval' must be specified in 'channel' mode.")
        
        # Convert minutes to ms and isolate the correct time point
        t_ms = t * 60000
        Y_data = data[data['time'] == t_ms - interval].drop(columns=['time'])
        
        # Transpose and log-transform Y matrix
        Y = np.log(Y_data.T.values)

    elif mode == 'apd90':
        # Log-transform scalar output and reshape for PLS
        Y = np.log(np.array(data).reshape(-1, 1))

    else:
        raise ValueError("Invalid mode. Use 'channel' or 'apd90'.")

    # Standardize input (X) and output (Y) matrices
    X_std = StandardScaler().fit_transform(X)
    Y_std = StandardScaler().fit_transform(Y)

    # Determine number of PLS components
    if n_components is None:
        n_components = min(X_std.shape[0], X_std.shape[1], 10)

    # Fit the PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_std, Y_std)

    # Extract and return sensitivity coefficients
    coefs = pls.coef_.flatten()
    return pd.DataFrame({'Parameter': param_names, 'Sensitivity': coefs})

def visualize_PoM_outputs(output_type='channel', n_repeats=5, subsample=20, base_path='Results', colors=None):
    """
    Visualizes PoM simulation results across repeats.

    Parameters:
    - output_type: str, one of ['channel', 'vm_10min', 'vm_20min']
    - n_repeats: int, number of repeats
    - subsample: int, index step to subsample simulations
    - base_path: str, folder where results are stored
    - colors: list of str, color for each repeat (optional)
    """

    if colors is None:
        colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = [f'Repeat {i+1}' for i in range(n_repeats)]

    plt.figure(figsize=(12, 6))

    for j in range(n_repeats):
        if output_type == 'channel':
            df = pd.read_csv(f'{base_path}/PoM_trafficking_M_repeat{j+1}.csv')
            time = df['time'] / 60000  # ms to min
            sims = df.drop(columns=['time'])

            # Subsample sims
            sims = sims.iloc[:, ::subsample]

            for col in sims.columns:
                plt.plot(time, sims[col], color=colors[j], alpha=0.15)

            mean_trace = sims.mean(axis=1)
            plt.plot(time, mean_trace, color=colors[j], label=labels[j], linewidth=2)

            plt.axvline(10, color='black', linestyle='--', lw=1)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Membrane Channel Level')
            plt.title('Membrane Channels Over Time (All Repeats)')
        
        elif output_type == 'vm_10min':
            df = pd.read_csv(f'{base_path}/PoM_V_repeat{j+1}.csv')
            beat_df = df[(df['time'] >= 599000) & (df['time'] <= 600000)].copy()
            time_ms = beat_df['time'] - 599000
            sims = beat_df.drop(columns=['time']).iloc[:, ::subsample]

            for col in sims.columns:
                plt.plot(time_ms, sims[col], color=colors[j], alpha=0.15)

            mean_trace = sims.mean(axis=1)
            plt.plot(time_ms, mean_trace, color=colors[j], label=labels[j], linewidth=2)

            plt.xlabel('Time (ms, last beat before 10 min)')
            plt.title('Membrane Potential at Last Beat Before 10 Minutes')

        elif output_type == 'vm_20min':
            df = pd.read_csv(f'{base_path}/PoM_V_repeat{j+1}.csv')
            beat_df = df[(df['time'] >= 1199800) & (df['time'] <= 1200000)].copy()
            time_ms = beat_df['time'] - 1199800
            sims = beat_df.drop(columns=['time']).iloc[:, ::subsample]

            for col in sims.columns:
                plt.plot(time_ms, sims[col], color=colors[j], alpha=0.15)

            mean_trace = sims.mean(axis=1)
            plt.plot(time_ms, mean_trace, color=colors[j], label=labels[j], linewidth=2)

            plt.xlabel('Time (ms, last beat before 20 min)')
            plt.title('Membrane Potential at Last Beat Before 20 Minutes')

        else:
            raise ValueError("output_type must be one of: 'channel', 'vm_10min', 'vm_20min'")

    plt.ylabel('Membrane Potential (mV)' if 'vm' in output_type else 'Level')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def drug_effects(freq, t, incub, interval, sk_block, ikur_block, ikr_block, k2p_block, AVE0118_conc, combined = False):
    """
    Simulates drug effects on cardiac electrophysiology under specified frequency and blocking conditions.

    Parameters:
        freq (float): Pacing frequency in Hz.
        t (float): Total simulation duration in minutes.
        incub (list): Pre-incubated model state to initialize from.
        interval (float): Logging interval in ms.
        sk_block (bool): Whether SK channel block is applied.
        ikur_block (bool): Whether IKur block is applied.
        ikr_block (bool): Whether IKr block is applied.
        k2p_block (bool): Whether k2p block is applied.
        AVE0118_conc (float): Concentration of AVE0118 for IKur block.
        combined (bool): Whether K2P and SK block are applied.

    Returns:
        dict: Simulation results with logged data, APD90, and final state.
    """
    # Load model
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

    # Calculate basic cycle length (ms)
    bcl = 1000 / freq

    # Define pacing protocol: 5 ms pulse every BCL, starting at 20 ms
    p = myokit.Protocol()
    p.schedule(1, 20, 5, bcl, 0)

    # Set up simulation
    s = myokit.Simulation(m, p)
    s.set_max_step_size(0.1)
    s.set_tolerance(1e-8, 1e-8)

    # Apply drug blocks
    if sk_block:
        s.set_constant('I_SK.sk_block', 0.2)
    if ikur_block:
        s.set_constant('parameters.AVE0118_conc', AVE0118_conc)
        s.set_constant('parameters.ikur_block', 1)
    if ikr_block:
        s.set_constant('I_Kr.ikr_block', 0.7)
    if k2p_block:
        s.set_constant('I_k2p.k2p_block', 0.2)
    if combined:
        s.set_constant('I_SK.sk_block', 0.2)
        s.set_constant('I_k2p.k2p_block', 0.2)

    # Set initial model state
    s.set_state(incub)

    # Compute total simulation time in ms
    sim_duration = t * 60000
    num_beats = sim_duration / bcl

    # Voltage threshold for APD90 (90% of diastolic V)
    vt = 0.9 * s.state()[m.get('membrane.V').index()]

    # Run the simulation and log results
    d, apd = s.run(bcl * num_beats, 
                   log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M'],
                   apd_variable='membrane.V',
                   apd_threshold=vt,
                   log_interval=interval)

    # Return results
    return dict(d=d, apd=apd, state=s.state())

def current_prepace_drugs(pp, freq, AVE0118_conc, sk_block=False, ikur_block=False, ikr_block=False, k2p_block=False, combined = False):
    """
    Pre-paces the model with optional drug blocks to reach steady-state before drug simulation.

    Parameters:
        pp (int): Number of pre-pace beats.
        freq (float): Pacing frequency in Hz.
        AVE0118_conc (float): Concentration of AVE0118 for IKur block.
        sk_block (bool): Whether SK channel block is applied.
        ikur_block (bool): Whether IKur block is applied.
        ikr_block (bool): Whether IKr block is applied.
        k2p_block (bool): Whether K2P block is applied.
        combined (bool): Whether K2P and SK block are applied.

    Returns:
        dict: Dictionary of initial states (with/without Ca_i and Vm components).
    """
    # Load model
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

    # Get variable indices
    cai_idx = m.get('Ca_Concentrations.Ca_i').index()
    v_idx = m.get('membrane.V').index()
    memv_idx = v_idx + 1  # To slice vector from membrane.V onward

    # Calculate BCL
    bcl = 1000 / freq

    # Set pacing protocol
    p = myokit.Protocol()
    p.schedule(1, 20, 5, bcl, 0)

    # Set up simulation
    s = myokit.Simulation(m, p)

    # Apply drug blocks
    if sk_block:
        s.set_constant('I_SK.sk_block', 0.2)
    if ikur_block:
        s.set_constant('parameters.AVE0118_conc', AVE0118_conc)
        s.set_constant('parameters.ikur_block', 1)
    if ikr_block:
        s.set_constant('I_Kr.ikr_block', 0.7)
    if k2p_block:
        s.set_constant('I_k2p.k2p_block', 0.2)
    if combined: 
        s.set_constant('I_SK.sk_block', 0.2)
        s.set_constant('I_k2p.k2p_block', 0.2)

    # Run pre-pacing
    s.pre(bcl * pp)

    # Get final state
    state_full = s.state()

    # Create variations of state
    state_no_cai = state_full.copy()
    del state_no_cai[cai_idx]

    state_vm = state_full[memv_idx:]
    state_vm_no_cai = state_vm.copy()
    del state_vm_no_cai[cai_idx]

    return dict(
        incub_comp=state_full,
        incub_cai=state_no_cai,
        incub_v=state_vm,
        incub_vcai=state_vm_no_cai
    )

def create_block_df_apd(apd10, apd20, sub = 1, offset=10):
    """
    Creates a DataFrame for APD90 values across two pacing frequencies (1 Hz and 5 Hz).

    Parameters
    ----------
    apd10 : dict
        Dictionary containing APD90 values and times for the first pacing period (e.g., 1 Hz for 10 min).
        Expected keys: 'time', 'apd90'.
    apd20 : dict
        Dictionary containing APD90 values and times for the second pacing period (e.g., 5 Hz for 10 min).
        Expected keys: 'time', 'apd90'.
    offset : float, optional
        Time offset (in minutes) to apply to the second time series (default is 10).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with columns ['Time', 'APD90', 'Freq'], indicating frequency and APD90 over time.
    """
    df1 = pd.DataFrame({
        'Time': apd10['time'],
        'APD90': apd10['apd90'],
        'Freq': 1
    })

    df2 = pd.DataFrame({
        'Time': apd20['time'][::sub] + offset,  # Subsampled and offset to match experimental design
        'APD90': apd20['apd90'][::sub],
        'Freq': 5
    })

    return pd.concat([df1, df2], ignore_index=True)


def create_block_df(data_10min, data_20min, sub = 1, offset=10):
    """
    Creates a DataFrame for membrane voltage (Vm) across two pacing frequencies (1 Hz and 5 Hz).

    Parameters
    ----------
    data_10min : dict
        Dictionary containing simulation results at 1 Hz for 10 minutes.
        Expected structure: {'d': {'engine.time': [...], 'membrane.V': [...]}}.
    data_20min : dict
        Dictionary containing simulation results at 5 Hz for 10 minutes.
        Expected structure: {'d': {'engine.time': [...], 'membrane.V': [...]}}.
    offset : float, optional
        Time offset (in minutes) to apply to the second time series (default is 10).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with columns ['Time', 'Vm', 'Freq'], indicating frequency and voltage trace over time.
    """
    df1 = pd.DataFrame({
        'Time': data_10min['d']['engine.time'],
        'Vm': data_10min['d']['membrane.V'],
        'Freq': 1
    })

    df2 = pd.DataFrame({
        'Time': np.array(data_20min['d']['engine.time'])[::sub] + offset,
        'Vm': np.array(data_20min['d']['membrane.V'])[::sub],
        'Freq': 5
    })

    return pd.concat([df1, df2], ignore_index=True)
