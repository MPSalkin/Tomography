
import os.path
import numpy as np

# Prompt user for input
Iterations = raw_input("How many iterations is the file? \n")
Subsets = raw_input("What is the number of subsets to be removed? \n")

# Open file for specified iterations
fname = 'Visuals/Data/Objective_Function_data_' + (Iterations) + 'Iterations.npy'
Data = np.load(fname)
Data = Data[()]

# Delete subsets from dictionary
del Data['Itr'+Subsets]
del Data['SSIM'+Subsets]
del Data['Time'+Subsets]

# Resave the data
np.save(fname, Data) 
