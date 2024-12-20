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

#exp type and Input_Variations are handled at the bottom of the main file "Experiment_Bremmer.py"
def run_optimizer(Exp_Type, Model_Variations):

	#basic variables
	duration = loadedParam["t_end"]
	trials = loadedParam["trials"]
	stim_pos = loadedParam["stim_pos"]  #old = ep_to_opt
	target_ep_t_offsets = loadedParam["target_ep_t_offsets"]
	batch_size = loadedParam["Batch_Size"]
	Epochs = loadedParam["Epochs"]
	if Exp_Type == "Random": 
	    Activation_Function = "linear"
	else:
	    Activation_Function = "ReLU"

	#loading Target and Input Data
	Target = np.load("../data/Data_" +str(Exp_Type)+ "/Ep_timelines_sequential_trials"+str(stim_pos)+".npy")
	Input_Training = np.load("../data/Data_" +str(Exp_Type)+ "/both_LIP_sequential_trials"+str(stim_pos)+".npy")
	Input_Prediction = np.load("../data/Data_" +str(Exp_Type)+ "/both_LIP_sequential_trials"+str(stim_pos)+".npy")
	if Exp_Type == "Random": Input_Prediction = np.load("../data/Data_Baseline/both_LIP_sequential_trials"+str(stim_pos)+".npy")
	
	
	
	#setup and learn networks for the scenarios of both maps presented, or only one of each.
	if Model_Variations == "both_LIP" or Model_Variations == "All":
		Input_both_LIP = Input_Training
		Input_layer_both_LIP =  keras.Input(shape=(40,40,2))
		flatt_both_LIP = keras.layers.Flatten()(Input_layer_both_LIP)
		Output_layer_both_LIP =keras.layers.Dense(3,activation= Activation_Function, use_bias = False)(flatt_both_LIP)
		
		
		
		if loadedParam["Create_New_Model"]:
			model_both_LIP = keras.Model(inputs=Input_layer_both_LIP, outputs=Output_layer_both_LIP)
			model_both_LIP.compile(optimizer="Adadelta", loss=keras.losses.MeanSquaredError())

		else:
			model_both_LIP = keras.models.load_model("../data/Data_" +str( Exp_Type )+ "/Model_both_LIP_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")



		#run and save
		history_both_LIP = model_both_LIP.fit(Input_both_LIP,Target,batch_size= batch_size, epochs=Epochs)
		weights_both_LIP = model_both_LIP.get_weights()
		prediction_both_LIP = model_both_LIP.predict(Input_Prediction)
		
		model_both_LIP.save("../data/Data_" +str( Exp_Type )+ "/Model_both_LIP_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")
		np.save("../data/Data_" +str( Exp_Type )+ "/prediction_both_LIP_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",prediction_both_LIP)
		np.save("../data/Data_" +str( Exp_Type )+ "/weights_both_LIP_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",weights_both_LIP)

			
		
		
		
	if Model_Variations == "only_Pc" or Model_Variations == "All":
		Input_only_Pc = Input_Training[:,:,:,0]
		Input_layer_only_Pc = keras.Input(shape=(40,40))
		flatt_only_Pc = keras.layers.Flatten()(Input_layer_only_Pc)
		Output_layer_only_Pc =keras.layers.Dense(3,activation= Activation_Function, use_bias = False)(flatt_only_Pc)
		
		if loadedParam["Create_New_Model"]:
			model_only_Pc = keras.Model(inputs=Input_layer_only_Pc, outputs=Output_layer_only_Pc)
			model_only_Pc.compile(optimizer="Adadelta", loss=keras.losses.MeanSquaredError())

		else:
			model_only_Pc = keras.models.load_model("../data/Data_" +str( Exp_Type )+ "/Model_only_Pc_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")



		#run and save 
		history_only_Pc = model_only_Pc.fit(Input_only_Pc,Target,batch_size= batch_size, epochs=Epochs)
		weights_only_Pc = model_only_Pc.get_weights()
		prediction_only_Pc  = model_only_Pc.predict(Input_Prediction[:,:,:,0])
		
		model_only_Pc.save("../data/Data_" +str( Exp_Type )+ "/Model_only_Pc_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")
		np.save("../data/Data_" +str( Exp_Type )+ "/prediction_only_Pc_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",prediction_only_Pc)
		np.save("../data/Data_" +str( Exp_Type )+ "/weights_only_Pc_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",weights_only_Pc)
		
		
		
		
	if Model_Variations == "only_Cd" or Model_Variations == "All":
		Input_only_Cd = Input_Training[:,:,:,1]
		Input_layer_only_Cd = keras.Input(shape=(40,40))
		flatt_only_Cd = keras.layers.Flatten()(Input_layer_only_Cd)
		Output_layer_only_Cd =keras.layers.Dense(3,activation= Activation_Function, use_bias = False)(flatt_only_Cd)
		
		if loadedParam["Create_New_Model"]:
			model_only_Cd = keras.Model(inputs=Input_layer_only_Cd, outputs=Output_layer_only_Cd)
			model_only_Cd.compile(optimizer="Adadelta", loss=keras.losses.MeanSquaredError())

		else:
			model_only_Cd = keras.models.load_model("../data/Data_" +str( Exp_Type )+ "/Model_only_Cd_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")
			

		#run and save model and weights
		history_only_Cd = model_only_Cd.fit(Input_only_Cd,Target,batch_size= batch_size, epochs=Epochs)
		weights_only_Cd = model_only_Cd.get_weights()
		prediction_only_Cd  = model_only_Cd.predict(Input_Prediction[:,:,:,1])
		
		model_only_Cd.save("../data/Data_" +str( Exp_Type )+ "/Model_only_Cd_offset_" + str(loadedParam["Sac_Start_Offset"])+".h5")
		np.save("../data/Data_" +str( Exp_Type )+ "/prediction_only_Cd_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",prediction_only_Cd)
		np.save("../data/Data_" +str( Exp_Type )+ "/weights_only_Cd_offset_" + str(loadedParam["Sac_Start_Offset"])+".npy",weights_only_Cd)
		
	


if __name__ == "_main_":
	Exp_Type = "Baseline"
	Model_Variations= "All"
	run_optimizer(Exp_Type,Model_Variations)


