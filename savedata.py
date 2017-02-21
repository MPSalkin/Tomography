import numpy as np
import os.path
import json
import csv

def saveme(objective, Time, SSIM, Iterations, SubSets, Algorithm):
	fname = 'Data/Objective_Function_data_' + str(Iterations) + 'Iterations.npy'

	if os.path.exists(fname):
		print 'Data file', fname, 'exists.'
	else:
		print 'Creating empty data file', fname,'to write to.'
		#A = np.zeros((8,Iterations))
		DataDict = {}
		# np.savetxt(fname, A)
		# with open(fname, 'w') as fp:
		# 	json.dump(DataDict, fp)
		np.save(fname, DataDict) 

 
# dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}
# w = csv.writer(open(fname, "w"))
# for key, val in dict.items():
# w.writerow([key, val])


	# with open(fname, 'r') as fp:
	# 	DataDict = json.load(fp)
	DataDict = np.load(fname)
	DataDict = DataDict[()]
	if SubSets == 0:
		DataDict['Itr0'] = objective
		DataDict['Time0'] = Time
		DataDict['SSIM0'] = SSIM
	else:
		DataDict['Itr'+str(SubSets)] = objective
		DataDict['Time'+str(SubSets)] = Time
		DataDict['SSIM'+str(SubSets)] = SSIM
	
	np.save(fname, DataDict) 

	# # Save
	# with open(fname, 'w') as fp:
	# 	json.dump(DataDict, fp)

	# Load
	# read_dictionary = np.load(fname).item()
	# print(read_dictionary[0][0]) # displays "world"


	# file = open(fname,'r')
	# Data = np.loadtxt(file)
	
	# obj = 0
	# t = 1
	# sim = 2
	# ss = 3
	# if Algorithm == 'FISTA':
	# 	Data[obj][:] = np.transpose(objective)
	# 	Data[t][:] = np.transpose(Time)
	# 	Data[sim][:] = np.transpose(SSIM)

	# if Algorithm == 'OSTR':
	# 	Data[ss][0] = (Subsets)
	# 	Data[3][:] = np.transpose(objective)
	# 	Data[4][:] = np.transpose(Time)
	# 	Data[5][:] = np.transpose(SSIM)

	# np.savetxt(fname, Data)
	# return