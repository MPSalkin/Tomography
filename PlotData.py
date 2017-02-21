#Plot Script 
import matplotlib.pyplot as pp
import numpy as np
import math
import os.path

Iterations = raw_input("How many iterations would you like to plot? \n")

fname = 'Data/Objective_Function_data_' + (Iterations) + 'Iterations.npy'

if os.path.exists(fname):
	print 'Data file', fname, 'exists.'
else:
	print 'Data file', fname,'does not exist. Try again.'
	quit()

file = open(fname,'r')
Data = np.load(file)
Data = Data[()]

# Plot Objective Function
for key in Data:
	if key[0] == 'T':
		time = Data[key]
		sub = key[4:]
		if key[-1] == 0:
			subset = 'FISTA'
		else:
			subset = 'OSTR ' + key[4:] + ' Subsets'		
		objective = Data['Itr'+sub]
		pp.figure(1)
		pp.plot(time,objective,label = subset)
		pp.title('Plot of objective function vs time - ' + Iterations +' Iterations')
		pp.ylabel('objective function')
		pp.xlabel('Time (seconds)')
pp.legend(loc = 'best')
pp.savefig('Figures/Objective_Function_vs_Time_' + Iterations + '_Iterations.png')
pp.show()

# Plot Objective Function Decrease
for key in Data:
	if key[0] == 'T':
		time = Data[key]
		sub = key[4:]
		if key[-1] == 0:
			subset = 'FISTA'
		else:
			subset = 'OSTR ' + key[4:] + ' Subsets'
		objective = Data['Itr'+sub]
		objective_decrease = objective[0] - objective
		pp.figure(2)
		pp.plot(time,objective_decrease,label = subset)
		pp.title('Plot of objective function decrease vs time - ' + Iterations +' Iterations')
		pp.ylabel('objective function decrease')
		pp.xlabel('Time (seconds)')
pp.legend(loc = 'best')
pp.savefig('Figures/Objective_Function_Decrease_vs_Time_' + Iterations + '_Iterations.png')
pp.show()

for key in Data:
	if key[0] == 'S':
		if key[-1] == 0:
			subset = 'FISTA'
		else:
			subset = 'OSTR ' + key[4:] + ' Subsets'
		SSIM = Data[key]
		pp.figure(3)
		pp.plot(SSIM,label = subset)
		pp.title('Plot of SSIM vs iteration - ' + Iterations +' Iterations')
		pp.ylabel('SSIM')
		pp.xlabel('Iteration')
pp.legend(loc = 'best')
pp.savefig('Figures/SSIM_vs_Iteration_' + Iterations + '_Iterations.png')
pp.show()
