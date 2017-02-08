import numpy as np
import os.path

def saveme(objective, Time, Iterations, Algorithm):
	fname = 'Data/Objective_Function_data_' + str(Iterations) + 'Iterations.txt'

	if os.path.exists(fname):
		print 'Data file', fname, 'exists.'
	else:
		print 'Creating empty data file', fname,'to write to.'
		A = np.zeros((4,Iterations))
		np.savetxt(fname, A)

	file = open(fname,'r')
	Data = np.loadtxt(file)
	
	if Algorithm == 'FISTA':
		Data[0][:] = np.transpose(objective)
		Data[1][:] = np.transpose(Time)
	if Algorithm == 'OSTR':
		Data[2][:] = np.transpose(objective)
		Data[3][:] = np.transpose(Time)
	np.savetxt(fname, Data)
	return