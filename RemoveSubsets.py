import os.path
import numpy as np

# Prompt user for input
Iterations = raw_input("How many iterations is the file? \n")
NumTot = int(raw_input('Enter the number of groups of subsets to be removed. \n'))
Subsets = [None]*(NumTot)
for i in range(0,(NumTot)):
	Subsets[i] = raw_input( str(i) + ": Enter subset value to be removed? \n")

# Open file for specified iterations
fname = 'Visuals/Data/Objective_Function_data_' + Iterations + 'Iterations.npy'
Data = np.load(fname)
Data = Data[()]

# Delete subsets from dictionary
for i in range(0,(NumTot)):
	del Data['Itr'+Subsets[i]]
	del Data['SSIM'+Subsets[i]]
	del Data['Time'+Subsets[i]]

# Resave the data
np.save(fname, Data) 
