#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 25/09/2024
"""
#%% Import packages and set directories.
import matplotlib
import matplotlib.pyplot as plt
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
import re
import os
import time
import multiprocessing

# Set directories. Note, the directories need to be set correctly on your own device.
work_dir = os.getcwd() 
work_dir = os.path.join(work_dir, 'SK_model')
os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

from SKTraffickingModelFunctions import incub_prot, current_prepace
#%% Perform incubation and pre-pace the model cellularly for different durations. 

# Load the model 
m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')

# Run the incubation protocol.
pp = 1000
current_pp1 = current_prepace(pp = pp, freq = 1, ca_i = None, sk_block = True)

# Run the protocol for 10 minutes and save the state.
def current_cell(freq, t, pp, ca_block = False, sk_block = True):
    
    # Load the model.
    m = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # Set the bcl.
    bcl = 1000/freq
    
    # Initalize  the protocol.
    p = myokit.Protocol()
   
    # Set the protocol.
    p.schedule(1, 20, 5, bcl, 0) 
    
    # Compile the simulation.
    s = myokit.Simulation(m, p)
    
    # Set the maximum timestep size and tolerances
    s.set_max_step_size(0.1)
    s.set_tolerance(1e-8, 1e-8)
        
    # Set the LTCC block if true.
    if ca_block is True:
        s.set_constant('I_Ca.ca_block', 0.3)
    if sk_block is True:
        s.set_constant('I_SK.sk_block', 0.2)
    
    # Set the incubated state as starting point.
    s.set_state(pp)
    
    # Determine the amount of beats for the duration of simulation.
    time_ms = t * 60000
    num_beats = time_ms/bcl
    
    # Run the simulation and calculate the APD90
    d = s.run(bcl * num_beats, log=['engine.time', 'membrane.V', 'I_SK.I_sk', 'I_SK_trafficking.M', 'I_SK_trafficking.S',
                                             'Ca_Concentrations.fTrap', 'Ca_Concentrations.Ca_sl', 'SR_Ca_Concentrations.Ca_sr',
                                             'Ca_Concentrations.Ca_i'], log_interval = 10)
    
    # Obtain the state.
    state = s.state()
        
    return dict(d = d, state = state)

curr_1hz_9min = current_cell(freq = 1, t = 9, pp = current_pp1['incub_comp'], ca_block = False, sk_block = True)
curr_1hz_10min = current_cell(freq = 1, t = 1, pp = curr_1hz_9min['state'], ca_block = False, sk_block = True)
curr_5hz_13min = current_cell(freq = 5, t = 3, pp = curr_1hz_10min['state'], ca_block = False, sk_block = True)
curr_5hz_16min = current_cell(freq = 5, t = 3, pp = curr_5hz_13min['state'], ca_block = False, sk_block = True)
curr_5hz_19min = current_cell(freq = 5, t = 3, pp = curr_5hz_16min['state'], ca_block = False, sk_block = True)
curr_5hz_20min = current_cell(freq = 5, t = 1, pp = curr_5hz_19min['state'], ca_block = False, sk_block = True)
curr_1hz_21min = current_cell(freq = 1, t = 1, pp = curr_5hz_20min['state'], ca_block = False, sk_block = True)
curr_1hz_22min = current_cell(freq = 1, t = 1, pp = curr_1hz_21min['state'], ca_block = False, sk_block = True)
curr_1hz_23min = current_cell(freq = 1, t = 1, pp = curr_1hz_22min['state'], ca_block = False, sk_block = True)
curr_1hz_30min = current_cell(freq = 1, t = 7, pp = curr_1hz_23min['state'], ca_block = False, sk_block = True)

# Offset the time for each segment to create a continuous timeline
def offset_time(data, start_min):
    time_offset = np.array(data['d']['engine.time']) / 60000  # Convert time to minutes
    return time_offset + start_min  # Add the start time offset

# Starting points in minutes for each segment
start_times = [10, 13, 16, 19, 20, 21]

# Define each time segment separately
min9_t = np.array(curr_1hz_9min['d']['engine.time']) / 60000
min10_t = np.array(curr_1hz_10min['d']['engine.time']) / 60000 + 9
min13_t = np.array(curr_5hz_13min['d']['engine.time']) / 60000 + 10
min16_t = np.array(curr_5hz_16min['d']['engine.time']) / 60000 + 13
min19_t = np.array(curr_5hz_19min['d']['engine.time']) / 60000 + 16
min20_t = np.array(curr_5hz_20min['d']['engine.time']) / 60000 + 19
min21_t = np.array(curr_1hz_21min['d']['engine.time']) / 60000 + 20
min22_t = np.array(curr_1hz_22min['d']['engine.time']) / 60000 + 21
min23_t = np.array(curr_1hz_23min['d']['engine.time']) / 60000 + 22
min30_t = np.array(curr_1hz_30min['d']['engine.time']) / 60000 + 23

# Concatenate all time segments into a continuous timeline
time_combined = np.concatenate([min10_t, min13_t, min16_t, min19_t, min20_t, min21_t, min22_t, min23_t, min30_t])

# Concatenate I_SK_trafficking.M and I_SK_trafficking.S values for the same segments
M_combined = np.concatenate([
    curr_1hz_9min['d']['I_SK_trafficking.M'],
    curr_1hz_10min['d']['I_SK_trafficking.M'],
    curr_5hz_13min['d']['I_SK_trafficking.M'], 
    curr_5hz_16min['d']['I_SK_trafficking.M'], 
    curr_5hz_19min['d']['I_SK_trafficking.M'], 
    curr_5hz_20min['d']['I_SK_trafficking.M'], 
    curr_1hz_21min['d']['I_SK_trafficking.M'],
    curr_1hz_22min['d']['I_SK_trafficking.M'],
    curr_1hz_23min['d']['I_SK_trafficking.M'],
    curr_1hz_30min['d']['I_SK_trafficking.M']])

ICa_combined = np.concatenate([
    curr_1hz_9min['d']['Ca_Concentrations.Ca_i'],
    curr_1hz_10min['d']['Ca_Concentrations.Ca_i'],
    curr_5hz_13min['d']['Ca_Concentrations.Ca_i'], 
    curr_5hz_16min['d']['Ca_Concentrations.Ca_i'], 
    curr_5hz_19min['d']['Ca_Concentrations.Ca_i'], 
    curr_5hz_20min['d']['Ca_Concentrations.Ca_i'], 
    curr_1hz_21min['d']['Ca_Concentrations.Ca_i'],
    curr_1hz_22min['d']['Ca_Concentrations.Ca_i'],
    curr_1hz_23min['d']['Ca_Concentrations.Ca_i'],
    curr_1hz_30min['d']['Ca_Concentrations.Ca_i']])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time_combined, ICa_combined, label='ICa', color='black')

# Add a vertical line at the 20-minute mark
plt.axvline(x=10, color='red', linestyle='--', label='5 Hz pacing starts')
plt.axvline(x=20, color='blue', linestyle='--', label='1 Hz pacing starts')

# Add labels, title, and legend
plt.xlabel('Time (minutes)')
plt.ylabel('Membrane channels (I_SK_trafficking.M)')
plt.title('Time-Dependent Development of ISK Membrane Channels')
plt.legend(loc='upper right')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Perform the 2D sims

# Create a pacing protocol.
def sims_2D(inputs):
    
    # Set default parameters.
    period = 13
    n = 600
    freq = 1
    dur = 100
    dur_sim = 1000
    conduct = 10
    interval = 5
    pp = curr_5hz_13min['state']
    sk_block = True

    # Initialize the model.
    model = myokit.load_model('MMT/Heijman_Trafficking_2023.mmt')
    
    # Calculate the bcl.
    bcl = int(1000/freq)
    
    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(bcl, 1, 0, 5, 0)
    s = myokit.SimulationOpenCL(model, p, ncells=[n, n])
    s.set_paced_cells(3, n, 0, 0)
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.001)
    
    # Set the constants.
    if sk_block is True:
        s.set_constant('I_SK.sk_block', 0.2)
    
    # Load the cellular pre-paced state.
    s.set_state(pp)
    
    # Run the model for the S1.
    if dur < inputs:
        log = s.run(inputs, log = ['membrane.V', 'engine.time', 'Ca_Concentrations.Ca_i'], log_interval=interval)
    else:
        log = s.run(dur, log = ['membrane.V', 'engine.time', 'Ca_Concentrations.Ca_i'], log_interval=interval)
    
    # Perform the simulation for 10 seconds
    for i in range(10):
        p2 = myokit.pacing.blocktrain(bcl, 1, inputs, 5, 1)
        s.set_protocol(p2)
        s.set_paced_cells(n/2, n/2, 0, 0)
    
        log = s.run(dur_sim, log=log, log_interval=interval)
        block = log.block2d()
        
        if sk_block is True:
            block.save(f'2D_sim_{period}min_{freq}hz_s1s2_{inputs}_skblock_nostop_ca.zip')
        else:
            block.save(f'2D_sim_{period}min_{freq}hz_s1s2_{inputs}_nostop_ca.zip')
        
    return dict(log=log, block=block)

#%% Multi-processing. 

if __name__ == '__main__':
    # Record the start time of the script execution.
    start_time = time.time()
    
    # Initialize an empty list to store the final results.
    final_results = []
    
    # Create a list of S1S2 values to iterate over.
    my_list = list(range(100, 180, 10))
    #my_list = [130]
    
    # Determine the number of CPU cores available and create a Pool object with a maximum number of processes.
    # This pool of processes will be used to perform parallel computations on the elements of 'my_list'.
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=min(num_processes, len(my_list)))
    
    # Apply the function to each element of 'my_list' in parallel.
    # The 'imap' method returns an iterator that yields the results of the function calls in the order of input.
    results = pool.imap(sims_2D, my_list)
    
    # Collect the results obtained from each iteration and store them in the 'final_results' list.
    # Also, save the simulation blocks obtained from each result to a file with a specific naming convention.
    for i, result in enumerate(results):
        final_results.append(result)
        result['block'].save(f'2D_sims_1Hz_13min_res{my_list[i]}_nostop_skblock_pt2.zip')

    # Close the Pool to prevent any more tasks from being submitted to it.
    pool.close()
    
    # Wait for all the worker processes to finish and terminate the Pool.
    pool.join()
    
    # Record the end time of the script execution.
    end_time = time.time()
    
    # Calculate the total time taken for the script execution.
    total_time_taken = end_time - start_time
    
    # If no error occurred during script execution, print the total time taken in seconds.
    print(f"Time taken: {total_time_taken} seconds")

#%% Calculate the reentry time.

def reentry_time_sk(c1, c2, s2, block, interval):
    """
    Analyzes a 2D matrix representing membrane potential data to determine 
    reentry time and depolarization events after a final S2 stimulus.

    Parameters:
    ----------
    c1 : float
        The voltage threshold for determining when the entire matrix is below this value (e.g., -50 mV).
    
    c2 : float
        The voltage threshold for identifying depolarization events (e.g., -65 mV).

    s2 : int
        The timing of the final S2 stimulus in the simulation (in milliseconds).
    
    block : myokit.DataBlock2d
        The data block containing the membrane potential data in a 2D matrix form (time, x, y).
    
    interval : int
        The time step interval in the simulation (in milliseconds).

    Returns:
    ----------
    pd.DataFrame
        A DataFrame containing the reentry time (rt) and the depolarization events (depol).
    """
    # Step 1: Determine the index corresponding to the final S2 stimulus.
    final_stim = int(s2 / interval + 1)

    # Step 2: Extract the membrane potential after the final S2 stimulus
    vm = block.get2d('membrane.V')
    vm_after_s2 = vm[final_stim:, :, :]  # Data after the final S2 stimulus

    time_steps = vm_after_s2.shape[0]  # Total number of time steps

    # Step 3: Find the first time point where the entire matrix is below c1 mV.
    below_cutoff_indices = np.where(np.all(vm_after_s2 < c1, axis=(1, 2)))[0]
    time_at_t = None

    if below_cutoff_indices.size > 0:
        # The first index where the entire matrix is below c1 mV.
        t = below_cutoff_indices[0]
        time_at_t = t * interval + final_stim * interval
        print(f"The matrix is entirely below {c1} mV at time point: {time_at_t} ms (relative to s2)")
    else:
        print(f"No time point found where the entire matrix is below {c1} mV.")

    # Step 4: Check for depolarization events after the matrix falls below c1.
    depolarization_events = []
    
    if time_at_t is not None:
        ongoing_event = False
        start_time = None
        
        # Loop through the time steps after the cutoff point.
        for t in range(below_cutoff_indices[0] + 1, time_steps):
            current_time = t * interval + final_stim * interval
            
            # If any part of the matrix exceeds c2, it's the start of a depolarization event.
            if np.any(vm_after_s2[t, :, :] > c2):
                if not ongoing_event:
                    start_time = current_time
                    ongoing_event = True  # Mark the start of an event
            else:
                # If the entire matrix drops below c2, mark the end of the depolarization event.
                if ongoing_event and np.all(vm_after_s2[t, :, :] < c2):
                    end_time = current_time
                    depolarization_events.append((start_time, end_time))
                    ongoing_event = False  # Reset for the next potential event

        # If an event was ongoing and didn't finish, record its end.
        if ongoing_event:
            end_time = (time_steps - 1) * interval + final_stim * interval
            depolarization_events.append((start_time, end_time))

        # Step 5: Report depolarization events.
        if depolarization_events:
            for event in depolarization_events:
                print(f"Depolarization event detected: Start time: {event[0]} ms, End time: {event[1]} ms")
        else:
            print("No depolarization events detected after the matrix fell below -50 mV.")
    
    # Return the results in a DataFrame.
    return pd.DataFrame(dict(rt=[time_at_t], depol=[depolarization_events]))

def process_files(directory, min_value, c1, c2, interval, skblock=False, no_stop=False):
    """
    Processes all files in a directory that match a specific filename pattern,
    applies the reentry_time_sk function to extract relevant data, and saves the
    results to CSV files, with optional inclusion of 'skblock' in the filenames.
    
    Parameters:
    ----------
    directory : str
        The directory containing the files to process.
    
    min_value : int
        The dynamic value for the minute portion in the filename, e.g., 23 in '2D_sim_23min_s1s2_150_skblock_nostop.zip'.
    
    c1 : float
        The voltage cutoff for determining when the matrix falls below a certain threshold (e.g., -50 mV).
    
    c2 : float
        The voltage cutoff for determining depolarization events (e.g., -65 mV).
    
    interval : int
        The time step interval used in the simulation (in milliseconds).
        
    skblock : boolean (default = False)
        Whether to include the 'skblock' tag in the output filenames and search pattern.
    
    no_stop : boolean (default = False)
        Whether the filename includes no_stop or not. 
    
    Returns:
    ----------
    None
        This function processes each file, extracts relevant data using the reentry_time_sk function, 
        and saves the resulting DataFrame as a CSV in the same directory.
    """
    
    # Create a dynamic regex pattern based on the min_value, skblock, and no_stop flags
    if skblock:
        if no_stop:
            pattern = fr'2D_sim_{min_value}min_1hz_s1s2_(\d+)_skblock_nostop\.zip'
        else:
            pattern = fr'2D_sim_{min_value}min_1hz_s1s2_(\d+)_skblock\.zip'
    else:
        if no_stop:
            pattern = fr'2D_sim_{min_value}min_1hz_s1s2_(\d+)_nostop\.zip'
        else:
            pattern = fr'2D_sim_{min_value}min_1hz_s1s2_(\d+)\.zip'
    
    for file in os.listdir(directory):
        # Match only files with the specific pattern, allowing dynamic min_value, skblock, and no_stop
        match = re.match(pattern, file)
        if match:
            s2 = int(match.group(1))
            block = myokit.DataBlock2d.load(os.path.join(directory, file))
            
            # Call reentry_time_sk and save results
            reentry_df = reentry_time_sk(c1, c2, s2, block, interval)
            
            # Determine output filename based on skblock and no_stop status
            if skblock:
                if no_stop:
                    output_filename = f'reentry_2D_sim_{min_value}min_1hz_s1s2_{s2}_skblock_nostop.csv'
                else:
                    output_filename = f'reentry_2D_sim_{min_value}min_1hz_s1s2_{s2}_skblock.csv'
            else:
                if no_stop:
                    output_filename = f'reentry_2D_sim_{min_value}min_1hz_s1s2_{s2}_nostop.csv'
                else:
                    output_filename = f'reentry_2D_sim_{min_value}min_1hz_s1s2_{s2}.csv'
                
            reentry_df.to_csv(os.path.join(directory, output_filename), index=False)
            print(f'Processed and saved: {output_filename}')
        else:
            print(f'Skipping file: {file}')

# Example call to loop through dictionary and output .csv files with the reentry time. 
process_files(work_dir, min_value=23, c1=-60, c2=-65, interval=5, skblock=True, no_stop=True)

#%% Calculate the DADs
def DAD_calc(data, dad, interval, minutes, s2, threshold, ticks, showit = True, diastole = True, prev = False):
    """
    Detects and visualizes Delayed Afterdepolarizations (DADs) from membrane potential data.

    This function analyzes post-reentry transmembrane voltage traces to identify DADs based on 
    threshold crossing. It calculates DAD amplitude, duration, and timing for each cell in the 
    2D spatial matrix, and optionally generates heatmap visualizations.

    Parameters
    ----------
    data : myokit.DataBlock2d
        Simulation data.
    
    dad : dict
        Output from the `reentry_time_sk` function; must contain a 'rt' key representing 
        the reentry termination time in ms.
    
    interval : int or float
        Simulation time step size in milliseconds (e.g., 5 ms).
    
    minutes : int
        Duration of simulation.
    
    s2 : int
        S2 interval (used in plot titles).
    
    threshold : float
        Voltage threshold (in mV) to define a depolarization event as a DAD (e.g., -65 mV).
    
    ticks : int
        Interval for tick marks in the generated heatmaps.
    
    showit : bool, optional (default=True)
        If True, displays heatmaps of the results.
    
    diastole : bool, optional (default=True)
        If True, the second heatmap will show time-to-threshold ("diastole"); 
        otherwise, it shows depolarization duration.
    
    prev : bool, optional (default=False)
        If True, visualizes the matrix at the timestep just before earliest DAD threshold crossing.

    Returns
    -------
    dict
        A dictionary containing:
        - 'amp' : 2D array of current amplitudes for each cell.
        - 'dur' : 2D array of depolarization durations.
        - 't1'  : 2D array of threshold crossing start times (in timesteps).
        - 't2'  : 2D array of depolarization end times (in timesteps).
        - 'dad' : int, number of DAD-like peaks at the center cell.

    Notes
    -----
    - Requires `find_peaks` from `scipy.signal`, `matplotlib.pyplot`, and `seaborn`.
    - Designed for post-reentry analysis where the matrix has returned to RMP.
    - Center cell (n//2, n//2) is used for initial peak detection to verify DAD presence.

    Example
    -------
    >>> block = myokit.DataBlock2d.load('2D_sim_19min_1hz_s1s2_130_skblock_nostop.zip')
    >>> dad_result = reentry_time_sk(c1=-60, c2=-65, s2=130, block=block, interval=5)
    >>> output = DAD_calc(data=block, dad=dad_result, interval=5, minutes=19,
                          s2=130, threshold=-65, ticks=20, showit=True)
    """
    
    # Load the membrane potential matrix.
    mat = data.get2d('membrane.V')
    
    # Determine the timepoint after which the entire matrix is back to rmp (post-reentry)
    rt = int(dad['rt'][0] / interval)
    
    # Reduce matrix size to only look for DADs after entire matrix has gone back to resting potential
    mat_red = mat[rt:, :, :]
    
    # Get the number of time steps and spatial dimensions (time, n, n)
    time_steps, n, _ = mat_red.shape
    
    # Select the middle element (assuming n is the same in both spatial dimensions)
    middle_element = mat_red[:, n//2, n//2]
    
    # Detect peaks that are above the threshold (e.g., -20 mV)
    peaks, _ = find_peaks(middle_element, height = -20)
    
    # Intiailize matrices to store the results.
    max_amp = np.full((n, n), np.nan)
    dur = np.full((n, n), np.nan)
    t1 = np.full((n, n), np.nan)
    t2 = np.full((n, n), np.nan)
    
    # Loop through each element in the matrix and record the first and last time 
    # the membrane potential passes the threshold together with the amplitude.
    for i in range(n):
        for j in range(n):
            # Get the membrane potential time series for the current element
            vm = mat_red[:, i, j]
            
            # Find the points where the vm passes the threshold
            above = np.where(vm > threshold)[0]
            
            # If there's a depolarization event
            if len(above) > 0:
                # Start of depolarization
                start = above[0]
                t1[i, j] = start
                
                # End of depolarization
                below = np.where(vm[start:] <= threshold)[0]
                
                if len(below) > 0:
                    end = start + below[0]
                    t2[i, j] = end
                    
                    # Calculate the duration of DAD
                    dur[i, j] = (end - start) * interval
                    
                    # Get the maximum potential during DAD
                    max_amp[i, j] = np.max(vm[start:end+1])
    
    # Find the earliest crossing time (ignorning NaNs)
    earliest = np.nanmin(t1)
    
    # Subset the matrix at the time point just before the earliest crossing
    previous = max(0, int(earliest) -1)  # Ensure the time is non-negative
    matrix_prev = mat_red[int(previous), :, :] 
    matrix_earliest = mat_red[int(earliest), :, :]
    
    if showit:
        if prev:
             # Set the figure size for two side-by-side heatmaps
             fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
             # Define tick positions 
             tick_positions = np.arange(0, max_amp.shape[0], ticks)
        
             # Create the heatmap for the amplitude at the previous beat( left)
             sns.heatmap(matrix_prev, cmap='plasma', annot=False, 
                         ax=axes[0], cbar_kws={'label': 'Max Amplitude (mV)'})
             axes[0].set_title(f'Max Amplitude at {int(previous * interval)} ms (cond = {minutes} min - s2 = {s2} ms)')
             axes[0].set_xlabel('X Coordinate')
             axes[0].set_ylabel('Y Coordinate')
             axes[0].set_xticks(tick_positions)
             axes[0].set_yticks(tick_positions)
        
             # Set the tick labels explicitly
             axes[0].set_xticklabels(tick_positions)
             axes[0].set_yticklabels(tick_positions)
             
             # Create the heatmap for the earliest beat crossing the threshold (right)
             sns.heatmap(matrix_earliest, cmap='plasma', annot=False, 
                         ax=axes[1], cbar_kws={'label': 'Max Amplitude (mV)'})
             axes[1].set_title(f'Max Amplitude at {int(earliest * interval)} ms (cond = {minutes} min - s2 = {s2} ms)')
             axes[1].set_xlabel('X Coordinate')
             axes[1].set_ylabel('Y Coordinate')
             axes[1].set_xticks(tick_positions)
             axes[1].set_yticks(tick_positions)
        
             # Set the tick labels explicitly for the right heatmap
             axes[1].set_xticklabels(tick_positions)
             axes[1].set_yticklabels(tick_positions)
        
             # Adjust layout to ensure everything fits well
             plt.tight_layout()
        
             # Show the plot
             plt.show()
     
        else:
            # Set the figure size for two side-by-side heatmaps
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
            # Define tick positions 
            tick_positions = np.arange(0, max_amp.shape[0], ticks)
    
            # Create the heatmap for max amplitude (left)
            sns.heatmap(max_amp, cmap='plasma', annot=False, 
                        ax=axes[0], cbar_kws={'label': 'Max Amplitude (mV)'})
            axes[0].set_title(f'Max Amplitude ({minutes} min - {s2} ms)')
            axes[0].set_xlabel('X Coordinate')
            axes[0].set_ylabel('Y Coordinate')
            axes[0].set_xticks(tick_positions)
            axes[0].set_yticks(tick_positions)
    
            # Set the tick labels explicitly
            axes[0].set_xticklabels(tick_positions)
            axes[0].set_yticklabels(tick_positions)
    
            # Create the heatmap for depolarization duration (right)
            if diastole:
                sns.heatmap(t1, cmap='plasma', annot=False, 
                            ax=axes[1], cbar_kws={'label': 'Diastole Duration (ms)'})
                axes[1].set_title(f'Diastole Duration ({minutes} min - {s2} ms)')
            else:    
                sns.heatmap(dur, cmap='plasma', annot=False, 
                            ax=axes[1], cbar_kws={'label': 'Depolarization Duration (ms)'})
                axes[1].set_title(f'Depolarization Duration ({minutes} min - {s2} ms)')
            axes[1].set_xlabel('X Coordinate')
            axes[1].set_ylabel('Y Coordinate')
            axes[1].set_xticks(tick_positions)
            axes[1].set_yticks(tick_positions)
    
            # Set the tick labels explicitly for the right heatmap
            axes[1].set_xticklabels(tick_positions)
            axes[1].set_yticklabels(tick_positions)
    
            # Adjust layout to ensure everything fits well
            plt.tight_layout()
    
            # Show the plot
            plt.show()
            
    return dict(amp = max_amp, dur = dur, t1 = t1, t2 = t2, dad = len(peaks))

# Example calculation DAD (can also manually go through the sims)
# First use the reentry_time_sk function to get the input for the DAD function.
block_19min_130_skblock = myokit.DataBlock2d.load('2D_sim_19min_1hz_s1s2_130_skblock_nostop.zip')
DAD_19min_130_skblock= reentry_time_sk(c1 = -60, c2 = -65, s2 = 130, block = block_19min_130_skblock, interval = 5)

# Run the DAD detector. 
min19_130_skb_DAD = DAD_calc(data = block_19min_130_skblock, dad = DAD_19min_130_skblock, interval = 5, minutes = 19, 
                s2 = 130, threshold = -65, ticks = 20, showit = False, diastole = True, prev = False)


