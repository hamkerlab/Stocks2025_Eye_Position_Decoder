"""
@author: nisto

Setup:
  - 1 fixation point and 1 saccade target
  - variable bar position where stimulus is flashed
"""



##############################
#### imports and settings ####
##############################
expType = "PLE_Barpos"
saveDir = '../data/PLE/barpos/flash_50ms/'
saveDirRates = saveDir+'rates/'
saveDirInterData = saveDir + 'interData/'
print("save at", saveDir)

import shutil
import numpy 
from pylab import *
import os 
from tqdm import trange
import Data_Processing as Data_Processing
import Plot_Results as Plot_Results
import Optimizer as Optimizer

# for default value
import sys
NO_STIM = sys.float_info.max

import numpy as np

import Modelparams
global loadedParam
loadedParam = Modelparams.defParams



# Including ANNarchy
import ANNarchy as ANN

# Including the network
from network_as_function import network_setup

# Including the world
from world import init_inputsignals, set_input



def precalcEvents(stim_pos):
    '''
    define events (spatial and temporal) representing setup
    events are: fixation, saccade, stimulus on- and offset

    return: dictionary with events and order according to event onset
    '''

    events = {}
    
    t_begin = loadedParam["t_begin"]
    t_sacon = loadedParam['t_sacon'] 
    t_target_stim = loadedParam["t_target_stim"]
    t_stimon = loadedParam['t_stimon'] 
    SacTarget = loadedParam['SacTarget']
    
    print(t_sacon)
    



    num_events = 5
    events['num_events'] = num_events
    # fixation
    event000 = {'name': 'Start_EP', 'stim': '', 'type': 'EVENT_EYEPOS',
                'time': t_begin, 'value': loadedParam['Fixation']}
    events['EVT000'] = event000

    #  Stimulus on
    event001 = {'name': 'Fixation_Stim', 'stim': 'stim', 'type': 'EVENT_STIMULUS',
                'time':t_stimon , 'value': loadedParam['Fixation']}
    events['EVT001'] = event001

    
    #  stimulus off
    event002 = {'name': 'Fixation_Stim', 'stim': 'stim', 'type': 'EVENT_STIMULUS',
                'time': t_target_stim, 'value': NO_STIM}
    events['EVT002'] = event002
    

    # Target Stimulus on
    event003 = {'name': 'Target_Stim', 'stim': 'stim', 'type': 'EVENT_STIMULUS',
                'time': t_target_stim, 'value': stim_pos}
    events['EVT003'] = event003

    # saccade
    event004 = {'name': 'Saccade', 'stim': '', 'type': 'EVENT_SACCADE',
                'time': t_sacon, 'value': SacTarget}
    events['EVT004'] = event004

    #get order by sorting events according to time
    eventtimes = []
    eventnames = []
    for i in range(num_events):
        eventtimes.append(events['EVT00' + str(i)]['time'])
        eventnames.append('EVT00' + str(i))
    order = sorted(range(num_events), key=lambda x: eventtimes[x])
    eventsOrderedByTime = []
    for i in range(num_events):
        eventsOrderedByTime.append(eventnames[order[i]])
    events['order'] = eventsOrderedByTime
    return events

##############
#### main ####
##############

def simulation(idx, net):

    # Calculate some needed init parameters
    duration = loadedParam['t_end']


    precalcParam = {}
    stim_pos = loadedParam["stim_pos"]
    # events
    precalcParam['EVENTS'] = precalcEvents(stim_pos)

    #init recording monitors
    
    Xr_pop = net.get_population('Xr')
    Xr_Monitor = ANN.Monitor(Xr_pop, "r")
    PC_pop = net.get_population('Xe_PC')
    PC_Monitor = ANN.Monitor(PC_pop, "r")
    CD_pop = net.get_population('Xe_CD')
    CD_Monitor = ANN.Monitor(CD_pop, "r")
    LIP_CD_pop = net.get_population('Xb_CD')
    LIP_CD_Monitor = ANN.Monitor(LIP_CD_pop,["r", "sum(FF1)","sum(FF2)", "sum(FB)"])
    LIP_PC_pop = net.get_population('Xb_PC')
    LIP_PC_Monitor = ANN.Monitor(LIP_PC_pop,["r", "sum(FF1)","sum(FF2)", "sum(FB)", "sum(exc)", "sum(inh)" ])
    FEF_pop = net.get_population('Xe_FEF')
    FEF_Monitor = ANN.Monitor(FEF_pop, ["r", "sum(FF1)","sum(FF2)"])
    Xh_pop = net.get_population('Xh')
    Xh_Monitor = ANN.Monitor(Xh_pop, "r")
    

    count = 0


    subdirname = str(stim_pos) + '_'
    signals = init_inputsignals(precalcParam,loadedParam,stim_pos,current_run, count, subdirname)


    ## Run the simulation for*duration* ms ##
    for t in range(duration):
        set_input(t, signals, net.get_populations())


        net.step()


    ## Save results ##
    # get recorded firing rates
    recorded_rates = {}

    # firing rates of neurons over time
    recorded_rates["Xr"] = Xr_Monitor.get("r", reshape=True)
    recorded_rates['Xe_PC'] =  PC_Monitor.get("r", reshape=True)
    recorded_rates['Xe_CD'] =  CD_Monitor.get("r", reshape=True)
    recorded_rates["Xb_CD"] =  LIP_CD_Monitor.get('r', reshape=True)
    recorded_rates['Xb_PC'] =  LIP_PC_Monitor.get("r", reshape=True)
    recorded_rates['FEF'] = FEF_Monitor.get("r", reshape=True)
    recorded_rates['Xh'] =  Xh_Monitor.get("r", reshape=True)
    

    # File paths and data map dynamically using variables
    file_data_map = {
    f"../data/Data_{Exp_Type}/Xr_{stim_pos}_{current_run}.npy": recorded_rates["Xr"],
    f"../data/Data_{Exp_Type}/PC_{stim_pos}_{current_run}.npy": recorded_rates['Xe_PC'],
    f"../data/Data_{Exp_Type}/CD_{stim_pos}_{current_run}.npy": recorded_rates['Xe_CD'],
    f"../data/Data_{Exp_Type}/LIP_CD_{stim_pos}_{current_run}.npy": recorded_rates["Xb_CD"],
    f"../data/Data_{Exp_Type}/LIP_PC_{stim_pos}_{current_run}.npy": recorded_rates["Xb_PC"],
    f"../data/Data_{Exp_Type}/FEF{stim_pos}_{current_run}.npy": recorded_rates["FEF"],
    f"../data/Data_{Exp_Type}/Xh_{stim_pos}_{current_run}.npy": recorded_rates["Xh"]
    }

    # Ensure directories exist and save files
    for file_path, data in file_data_map.items():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        np.save(file_path, data)
        print(f"Data saved successfully to {file_path}")

    net.reset(True, False)


    
    

    return


###########################################################################


def generate_data(Exp_Type):
    ## Simulation ##
    network_setup(loadedParam)
    sac_samples = np.load("./Saccades.npy")
    

    
    global current_run

    for i in trange(loadedParam["trials"]):
        network_setup(loadedParam)
        current_run=i

        loadedParam["t_sacon"] = loadedParam["t_target_stim"] + loadedParam["Sac_Start_Offset"]   + int(sac_samples[i]) 
 
        

        Max_processes = min(42, 1)
        
        
        results = ANN.parallel_run(method=simulation, number=1, max_processes=Max_processes)#, annarchy_json = "test.json")

        
if __name__ == '__main__':

    
    global Exp_Type
    
    
    Exp_Type = loadedParam["Experiment"]

        


    
    #input variations : "All", "both_LIP", "only_Pc", "only_Cd"
    #"All" is the slowest Option because it optimizes the networks in sequence. To increase speed, start seperate screens using "Both_LIP", "Only_PC", and "Only_Cd".
    Model_Variations = "All"




    generate_data(Exp_Type)
    Data_Processing.data_preprocessing(Exp_Type)
    Optimizer.run_optimizer(Exp_Type,Model_Variations)
    Data_Processing.opt_data_post_processing(Exp_Type,Model_Variations)
    Plot_Results.Plot_Xu_GFI_Median(Exp_Type)
    Plot_Results.Plot_Aligned_Images(Exp_Type, Model_Variations)
    
   
 
