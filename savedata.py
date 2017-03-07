import numpy as np
import os.path

def saveme(objective, Time, SSIM, Iterations, SubSets):
	fname = 'Visuals/Data/Objective_Function_data_' + str(Iterations) + 'Iterations.npy'

	if os.path.exists(fname):
		print 'Data file', fname, 'exists.'
	else:
		print 'Creating empty data file', fname,'to write to.'
		DataDict = {}
		np.save(fname, DataDict) 

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
