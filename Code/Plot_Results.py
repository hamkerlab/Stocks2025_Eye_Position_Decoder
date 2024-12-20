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
from matplotlib.colors import LinearSegmentedColormap#
import os

global loadedParam
import Modelparams
loadedParam = Modelparams.defParams


            
def Plot_Xu_GFI_Median(Exp_Type):

    if Exp_Type == "Random": print("This isnt really meant to work with the Random Setting")
    stim_pos = 20   #hardcoded for now, but its basially always 20
    trials = loadedParam['trials']
    offset = 262 
    time = np.linspace(300, 700, 400, dtype=int)
    time_relativeToSaccadeOffset = time - offset 
    flashes = [50,150,250,350]
    flashes_corrected= [50-38,150-38,250-38,350-38]
    Xu_High_to_Low = [0.915, 0.825, 0.02, 0.0]
    Xu_Low_to_High = [0.935, 0.74, 0.0, 0.0]
    
    
    
    Individual_LIP = np.load("../data/Data_" +str(Exp_Type)+ "/Individual_Aligned_trials"+str(stim_pos)+".npy",allow_pickle=True)
    PC_Xu_Cross_Index_Vector = np.zeros((400,15*trials,2))
    Probe_HL = np.zeros((400,15*trials))
    Probe_LH = np.zeros((400,15*trials))
    PC_Xu_Cross_Index_Vector_Average = np.zeros((400,2))
    PC_Xu_Cross_Index_Vector_Median = np.zeros((400,2))
    
    Presteady_HL = np.reshape(Individual_LIP[:,300,19:22,18:23,0],(15*trials))
    Poststeady_HL = np.reshape(Individual_LIP[:,680,19:22,18:23,0],(15*trials))
    Presteady_LH = np.reshape(Individual_LIP[:,300,19:22,28:33,0],(15*trials))
    Poststeady_LH = np.reshape(Individual_LIP[:,680,19:22,28:33,0],(15*trials))

    for t in range(400):
        n= t+ 300
        Probe_HL[t] = np.reshape(Individual_LIP[:,n,19:22,18:23,0],(15*trials))
        PC_Xu_Cross_Index_Vector[t,:,0] = ( Probe_HL[t] - Poststeady_HL)/ (Presteady_HL - Poststeady_HL)
        PC_Xu_Cross_Index_Vector_Median[t,0] = np.median(PC_Xu_Cross_Index_Vector[t,:,0])
        PC_Xu_Cross_Index_Vector_Average[t,0] = np.average(PC_Xu_Cross_Index_Vector[t,:,0])
        
        Probe_LH[t] = np.reshape(Individual_LIP[:,n,19:22,28:33,0],(15*trials))
        PC_Xu_Cross_Index_Vector[t,:,1] = ( Probe_LH[t] - Poststeady_LH)/ (Presteady_LH - Poststeady_LH)
        PC_Xu_Cross_Index_Vector_Median[t,1] = np.median(PC_Xu_Cross_Index_Vector[t,:,1])
        PC_Xu_Cross_Index_Vector_Average[t,1] = np.average(PC_Xu_Cross_Index_Vector[t,:,1])
        
    
    
    plt.subplot()
    plt.plot(flashes,PC_Xu_Cross_Index_Vector_Median[flashes_corrected,0],'-o',label= "GFI-Median")
    plt.plot(flashes,Xu_High_to_Low,'-^',label= "Xu GFI-Median")
    plt.xlabel("Time relative to Saccade Offset (ms)",fontsize = 20)
    plt.ylabel("GFI High-to-Low",fontsize = 20)
    plt.xticks(flashes, fontsize=20)
    plt.yticks([0,0.5,1], fontsize=20)
    plt.legend(fontsize =17)
    plt.savefig("../data/" + str( Exp_Type )+"_Plots/GFI_Median_High_to_Low.svg")
    plt.close()
    
    plt.subplot()
    plt.plot(flashes,PC_Xu_Cross_Index_Vector_Median[flashes_corrected,1],'-o',label= "GFI-Median")
    plt.plot(flashes,Xu_Low_to_High,'-^',label= "Xu GFI-Median")
    plt.xlabel("Time relative to Saccade Offset (ms)",fontsize = 20)
    plt.ylabel("GFI Low-to-High",fontsize = 20)
    plt.xticks(flashes, fontsize=20)
    plt.yticks([0,0.5,1], fontsize=20)
    plt.legend(fontsize =17)
    
    plt.savefig("../data/" + str( Exp_Type )+"_Plots/GFI_Median_Low_to_High.svg")
    plt.close()
            
            
            
   
def Plot_Aligned_Images(Exp_Type, Model_Variations):

    #basic variables
    duration = loadedParam["t_end"]
    trials = loadedParam["trials"]
    stim_pos = loadedParam["stim_pos"]  #old = ep_to_opt
    
    Slice_Size = loadedParam["Aligned_Sac_right_border"] - loadedParam["Aligned_Sac_left_border"]
    target_ep_t_offsets = loadedParam["target_ep_t_offsets"]
    time = np.linspace(0, Slice_Size, Slice_Size)
    time_relativeToSaccadeOnset = time + loadedParam["Aligned_Sac_left_border"]
    important_timesteps = [150,200,250,300,350,400,450,500,550,600]
    
    #loading Target, input, prediction and weights
    Centered_Target = np.load("../data/Data_Baseline/Centered_to_Sac_Onset_EP_timelines"+str(stim_pos)+".npy",allow_pickle=True)
    Centered_Input= np.load("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_LIP"+str(stim_pos)+".npy",allow_pickle=True)
    #loading dictionaries sorted by model model_variations. .item() necessary to process dictionaries with np.load
    Centered_Prediction = np.load("../data/Data_" +str(Exp_Type)+ "/Centered_to_Sac_Onset_prediction"+str(stim_pos)+".npy",allow_pickle=True)
    Centered_Prediction = Centered_Prediction.item()
    Weights = np.load("../data/Data_" +str(Exp_Type)+ "/reshaped_weights"+str(stim_pos)+".npy",allow_pickle=True)
    Weights = Weights.item()
    

    

    
####Gigantic Weight saving Part, written in duplicate to create versions without colorbars and axis labels

    save_dir = f"../data/{Exp_Type}_Plots/"
    os.makedirs(save_dir, exist_ok=True)
    if Model_Variations=="All":
        Model_Variations = ["both_LIP","only_Pc", "only_Cd"]
    else:
        Model_Variations = [Model_Variations]
        
        
    for variation in Model_Variations:
        Weights[variation] = np.swapaxes(Weights[variation],0,1)   #quickly turning the weight maps so they align visually with the inputs
        
        
        
    #save Images of Weight Maps
    for offset in range(len(target_ep_t_offsets)):    #only produces the weights for both maps currently, if the single map weights are relevant, you need to remove the dimension that sigifies the map

        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Weights["both_LIP"][:,:,0,offset],cmap = "bwr",vmin = -6, vmax = 6) 
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP PC Weights for Offset = "  + str(target_ep_t_offsets[offset]) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)  
        cbar = fig.colorbar(mappable)
        cbar.set_ticks([-6, 6])  
        cbar.set_ticklabels(["-6", "6"])  
        cbar.ax.tick_params(labelsize=20)  
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/WeightsPC_both_LIP_" + str(target_ep_t_offsets[offset])+".svg")
        ax.clear()
        
        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Weights["both_LIP"][:,:,1,offset],cmap = "bwr",vmin = -6, vmax = 6) 
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP CD Weights for Offset = "  + str(target_ep_t_offsets[offset]) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)  
        cbar = fig.colorbar(mappable)
        cbar.set_ticks([-6, 6])  
        cbar.set_ticklabels(["-6", "6"])  
        cbar.ax.tick_params(labelsize=20)
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/WeightsCD_both_LIP_" + str(target_ep_t_offsets[offset])+".svg")
        plt.close()
        
        
        
        
        #version without labels and cbar
        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Weights["both_LIP"][:,:,0,offset],cmap = "bwr",vmin = -6, vmax = 6) 
        #ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        #ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft = False)
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/WeightsPC_both_LIP_sparse_" + str(target_ep_t_offsets[offset])+".svg")
        ax.clear()
        
        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Weights["both_LIP"][:,:,1,offset],cmap = "bwr",vmin = -6, vmax = 6) 
        plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft = False)
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/WeightsCD_both_LIP_sparse_" + str(target_ep_t_offsets[offset])+".svg")
        plt.close()        
        
        
        
            
            
    #save images of LIP-Maps        
    for step in important_timesteps:
        sac_relative = step - 200
        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Centered_Input[step,:,:,0],cmap = "Reds",vmin = 0, vmax = 0.25)
        #re-set Axis Ticks for Maps
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP PC at t = " + str(sac_relative) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/LIP_PC_MAP" + str(sac_relative)+".svg")
        ax.clear()
        
        
                
        mappable = ax.imshow(Centered_Input[step,:,:,1],cmap = "Reds",vmin = 0, vmax = 0.25)
        #re-set Axis Ticks for Maps
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP CD at t = " + str(sac_relative) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)
        
        
        
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/LIP_CD_MAP" + str(sac_relative)+".svg")
        plt.close()
        
                
    for step in important_timesteps:
        sac_relative = step - 200
        fig,ax= plt.subplots()
        
        mappable = ax.imshow(Centered_Input[step,:,:,0],cmap = "Reds",vmin = 0, vmax = 0.25)
        #re-set Axis Ticks for Maps
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP PC at t = " + str(sac_relative) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)
        cbar = fig.colorbar(mappable)
        cbar.set_ticks([0, 0.25])  
        cbar.set_ticklabels(["0", "0.25"])  
        cbar.ax.tick_params(labelsize=20)  
        plt.savefig("../data/" + str( Exp_Type )+"_Plots/LIP_PC_MAP_Cbar" + str(sac_relative)+".svg")
        ax.clear()

        
        fig,ax= plt.subplots()        
        mappable = ax.imshow(Centered_Input[step,:,:,1],cmap = "Reds",vmin = 0, vmax = 0.25)
        #re-set Axis Ticks for Maps
        ax.set_xticks([0,10,20,30,39],fontsize = 20)  # Set the desired tick locations
        ax.set_xticklabels(["-40", "-20", "0","20","40"],fontsize = 20)  # Set the desired tick labels
        ax.set_yticks([39,30,20,10,0],fontsize = 20)  # Set the desired tick locations
        ax.set_yticklabels(["40", "20", "0","-20","-40"],fontsize = 20)  # Set the desired tick labels
        #ax.set_title("LIP CD at t = " + str(sac_relative) + "ms",fontsize = 20)
        ax.set_ylabel("Stimulus Position ($\degree$)",fontsize = 20)
        ax.set_xlabel("Eye Position ($\degree$)",fontsize = 20)
        
        cbar = fig.colorbar(mappable)
        cbar.set_ticks([0, 0.25])  
        cbar.set_ticklabels(["0", "0.25"])  
        cbar.ax.tick_params(labelsize=20)  

        plt.savefig("../data/" + str( Exp_Type )+"_Plots/LIP_CD_MAP_Cbar" + str(sac_relative)+".svg")
        plt.close()
        
        
    #save Images of Fits
    for variation in Model_Variations:
        for offset in range(len(target_ep_t_offsets)):
        
            fig,ax= plt.subplots()
            ax.plot(time_relativeToSaccadeOnset,Centered_Prediction[variation][:,offset],linewidth = 2, color = "red", label = "Model Prediction")
            ax.plot(time_relativeToSaccadeOnset,Centered_Target[:,offset],linewidth = 2,color = "black", label = "Target Eye Position")
            ax.set_yticks([0,5,10,15,20])
            ax.set_yticklabels([0,5,10,15,20],fontsize = 20)
            ax.set_xticks([-200,0,200,400])
            ax.set_xticklabels([-200,0,200,400],fontsize = 20)

            #ax0.set_title(Title)
            ax.set_xlabel("Time relative to  Saccade Onset (ms)",fontsize = 20)
            ax.set_ylabel("Eye Position ($\degree$)",fontsize = 20)
            #ax.set_title("Target:  Eye Position shifted by " + str(target_ep_t_offsets[offset])+ " ms")
            ax.legend(loc= "lower right", fontsize = 17)
            
            
            plt.savefig("../data/" + str( Exp_Type )+"_Plots/Fit_for_" + str(variation)+"_"+ str(target_ep_t_offsets[offset])+".svg")
            plt.close()
        




