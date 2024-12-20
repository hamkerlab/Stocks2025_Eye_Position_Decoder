from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.ticker as ticker


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as signal
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import lognorm
from matplotlib.colors import LinearSegmentedColormap

global loadedParam
import Modelparams
loadedParam = Modelparams.defParams


def data_preprocessing(Exp_Type):

    ######################## Settings ####################
    duration = loadedParam["t_end"]
    trials = loadedParam["trials"]
    stim_pos = loadedParam["stim_pos"] 
    target_ep_t_offsets = loadedParam["target_ep_t_offsets"]
    Noise = loadedParam["Noise_on_LIP"]
    Sac_Onsets = np.load("./Saccades.npy")
    Sac_Onsets_in_Simtime = Sac_Onsets + loadedParam["t_target_stim"] + loadedParam["Sac_Start_Offset"] 
    
    if loadedParam["Opt_for_Stim_or_EP"] == "Stim": Stimpos_timeline_all_trials = np.load("../data/Data_" +str(Exp_Type)+ "/Stimpos_timeline_all_trials.npy")



    #load EP and LIP Raw Data for all Trials
    Eye_Pos = {}
    LIP_PC = {}
    LIP_CD = {}

    for trial in range(trials):
        if loadedParam["Experiment"] == "Random": Eye_Pos[trial] =  np.loadtxt("../data/Eye_Pos/" + str(trial) + "_eyepos.txt")
        else: Eye_Pos[trial] = np.loadtxt("../data/Eye_Pos/" +str(stim_pos)+"_" + str(trial) + "_eyepos.txt")
        Eye_Pos[trial] = Eye_Pos[trial][:,1]
        LIP_PC[trial] = np.load("../data/Data_" +str(Exp_Type)+ "/LIP_PC_"+ str(stim_pos) +"_" + str(trial)+ ".npy")
        LIP_CD[trial] = np.load("../data/Data_" +str(Exp_Type)+ "/LIP_CD_"+ str(stim_pos) +"_" + str(trial)+ ".npy")
        
    
    #Create the Eye-Position Timelines of past present and future and create a single LIP-File
    Ep_timelines = np.zeros((trials, duration,3))
    both_LIP = np.zeros((trials,duration,40,40,2))
    for trial in range(trials):
        #-100
        Ep_timelines[trial,:,0]= np.roll(Eye_Pos[trial], target_ep_t_offsets[0])
        Ep_timelines[trial,(duration- abs(target_ep_t_offsets[0])):None,0] = Ep_timelines[trial,(duration- abs(target_ep_t_offsets[0]))-100,0]
        #0
        Ep_timelines[trial,:,1]= Eye_Pos[trial]
        #+200
        Ep_timelines[trial,:,2]= np.roll(Eye_Pos[trial], target_ep_t_offsets[2])
        Ep_timelines[trial,0:target_ep_t_offsets[2],2] = loadedParam["Fixation"]
        
        #create single LIP File
        both_LIP[trial,:,:,:,0] = LIP_PC[trial]
        both_LIP[trial,:,:,:,1] = LIP_CD[trial]
        
        
    #create single time axis to be compatible with Keras
    Ep_timelines_sequential_trials = np.reshape(Ep_timelines, (trials* duration, 3))
    both_LIP_sequential_trials = np.reshape(both_LIP, (trials* duration,40,40,2))
    if loadedParam["Opt_for_Stim_or_EP"] == "Stim": Stimpos_timeline_sequential_trials = np.reshape(Stimpos_timeline_all_trials, (trials*duration))
    
    #create Averages for use in Plotting Scrips
    Average_Ep_Timelines = np.average(Ep_timelines, axis = 0)
    Average_both_LIP = np.average(both_LIP, axis = 0)
    
    
    #Align Slice Ep and Individual Saccades, centered on the Saccade Onset. 
    #To keep this slice always the same size, the simulation time was set to a extremely large value. This is mostly done because simulation time wasnt a primary concern.
    #This is done "mostly" to keep the time axis of plots using this data as "neat" as possible.
    
    Slice_Range = loadedParam["Aligned_Sac_right_border"] - loadedParam["Aligned_Sac_left_border"]
    Centered_to_Sac_Onset_EP = np.zeros(Slice_Range)
    Centered_to_Sac_Onset_EP_timelines = np.zeros((Slice_Range,3))
    Centered_to_Sac_Onset_LIP = np.zeros((Slice_Range,40,40,2))
    Merged_LIP_to_Sac_Onset = np.zeros((trials,Slice_Range,40,40))
    Individual_Aligned_trials = np.zeros((trials,Slice_Range,40,40,2))
    
    for trial in range(trials):
        #just here to make the brackets more readable
        left_border = int(Sac_Onsets_in_Simtime[trial]+loadedParam["Aligned_Sac_left_border"])
        right_border = int(Sac_Onsets_in_Simtime[trial]+loadedParam["Aligned_Sac_right_border"])
        

        Centered_to_Sac_Onset_EP_timelines[:,:] += Ep_timelines[trial,left_border:right_border,:]
        Centered_to_Sac_Onset_EP += Eye_Pos[trial][left_border:right_border]
        Centered_to_Sac_Onset_LIP += both_LIP[trial,left_border:right_border,:,:,:]
        Individual_Aligned_trials[trial,:,:,:,:] = both_LIP[trial,left_border:right_border,:,:,:]
        Merged_LIP_to_Sac_Onset[trial,:,:,:] = 0.5*both_LIP[trial,left_border:right_border,:,:,0] + 0.45* both_LIP[trial,left_border:right_border,:,:,1] 
                
    Centered_to_Sac_Onset_EP /= trials
    Centered_to_Sac_Onset_EP_timelines /= trials
    Centered_to_Sac_Onset_LIP /=trials

    
    
    #save EP Datadata
    np.save("../data/Data_" +str(Exp_Type)+ "/Eye_Pos"+str(stim_pos)+".npy",Eye_Pos)
    np.save("../data/Data_" +str(Exp_Type)+ "/Ep_timelines"+str(stim_pos)+".npy",Ep_timelines)
    np.save("../data/Data_" +str(Exp_Type)+ "/Ep_timelines_sequential_trials"+str(stim_pos)+".npy",Ep_timelines_sequential_trials)
    np.save("../data/Data_" +str(Exp_Type)+ "/Average_Ep_Timelines"+str(stim_pos)+".npy",Average_Ep_Timelines)
    np.save("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_EP"+str(stim_pos)+".npy",Centered_to_Sac_Onset_EP)
    np.save("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_EP_timelines"+str(stim_pos)+".npy",Centered_to_Sac_Onset_EP_timelines)
    
    
    #save LIP Data
    np.save("../data/Data_" +str(Exp_Type)+ "/both_LIP"+str(stim_pos)+".npy",both_LIP)
    np.save("../data/Data_" +str(Exp_Type)+ "/both_LIP_sequential_trials"+str(stim_pos)+".npy",both_LIP_sequential_trials)
    np.save("../data/Data_" +str(Exp_Type)+ "/Average_both_LIP"+str(stim_pos)+".npy",Average_both_LIP)
    np.save("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_LIP"+str(stim_pos)+".npy",Centered_to_Sac_Onset_LIP)
    np.save("../data/Data_" +str(Exp_Type)+ "/Merged_LIP_to_Sac_Onset"+str(stim_pos)+".npy",Merged_LIP_to_Sac_Onset)
    np.save("../data/Data_" +str(Exp_Type)+ "/Individual_Aligned_trials"+str(stim_pos)+".npy",Individual_Aligned_trials)
    
    #save Stimpos Data
    if loadedParam["Opt_for_Stim_or_EP"] == "Stim":  np.save("../data/Data_" +str(Exp_Type)+ "/Stimpos_timeline_sequential_trials.npy",Stimpos_timeline_sequential_trials)
    
def opt_data_post_processing(Exp_Type,Model_Variations):
    
    if Model_Variations=="All":
        Model_Variations = ["both_LIP","only_Pc", "only_Cd"]
    
    else:
        Model_Variations = [Model_Variations]
        
    stim_pos = loadedParam["stim_pos"]  #old = ep_to_opt
    Slice_Range = loadedParam["Aligned_Sac_right_border"] - loadedParam["Aligned_Sac_left_border"]
    Sac_Onsets = np.load("./Saccades.npy")
    Sac_Onsets_in_Simtime = Sac_Onsets + loadedParam["t_target_stim"] + loadedParam["Sac_Start_Offset"] 
    
    
    
    #model_variations are done here here using dictionaries and a loop instead of the hard coded variation of the optimizer script.
    #you could do that in the optimizer script too, but since you need to change the geometry of numpy matrixes depending on the case, you would need to hard code geometry changes at some point
    #to translate the model variations from language to said geometries. I dont think that would have actually made things easier. For all variation changes that arent caught by the 3 hardcoded cases
    #one would need to re-run the network anyways since it would produce diffrent raw data, making that also somewhat moot.
    
    prediction = {}
    average_prediction = {}
    centered_to_Sac_Onset_prediction = {}
    weights = {}
    for variation in Model_Variations:
    #starting with the prediction
    
        prediction[variation] = np.load("../data/Data_" +str( Exp_Type )+ "/prediction_"+str(variation)+"_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy")
        prediction[variation] = np.reshape(prediction[variation],(loadedParam["trials"],loadedParam["t_end"],3))   #This line Breaks if the "Random" Variation of the setting uses a geometry of Runs and Timesteps that does not match the baseline setting
        average_prediction[variation] = np.average(prediction[variation], axis = 0)
        centered_to_Sac_Onset_prediction[variation] = np.zeros((Slice_Range,3))
        
        #aligning individual predictions to saccade start
        for trial in range(loadedParam["trials"]):
            #just here to make the brackets more readable
            left_border = int(Sac_Onsets_in_Simtime[trial]+loadedParam["Aligned_Sac_left_border"])
            right_border = int(Sac_Onsets_in_Simtime[trial]+loadedParam["Aligned_Sac_right_border"])
            centered_to_Sac_Onset_prediction[variation] += prediction[variation][trial,left_border:right_border,:]
        
        centered_to_Sac_Onset_prediction[variation] /= loadedParam["trials"]  #all predictions should now be sliced and aligned to the onset of the "real" saccade
        
        
        #now dealing with weights
        weights[variation] = np.load("../data/Data_" +str( Exp_Type )+ "/weights_"+str(variation)+"_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy")
        weights[variation] = weights[variation][0,:]
        if variation == "both_LIP":
            weights[variation] = np.reshape(weights[variation],(40,40,2,3))
        else:
            weights[variation] = np.reshape(weights[variation],(40,40,3))
        weights[variation] = np.swapaxes(weights[variation],0,1)
        
    #saving data
    np.save("../data/Data_" +str(Exp_Type)+ "/reshaped_prediction"+str(stim_pos)+".npy",prediction)
    np.save("../data/Data_" +str(Exp_Type)+ "/average_prediction"+str(stim_pos)+".npy",average_prediction)
    np.save("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_prediction"+str(stim_pos)+".npy",centered_to_Sac_Onset_prediction)
    np.save("../data/Data_" +str(Exp_Type)+ "/reshaped_weights"+str(stim_pos)+".npy",weights)





if __name__ == "__main__":
    Exp_Type = "Baseline"
    Model_Variations ="All"
    data_preprocessing(Exp_Type)
