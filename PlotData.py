#Plot Script 
import matplotlib.pyplot as pp
import numpy as np
import math
import os.path

Iterations = raw_input("How many iterations would you like to plot? \n")

fname = 'Data/Objective_Function_data_' + (Iterations) + 'Iterations.txt'

if os.path.exists(fname):
	print 'Data file', fname, 'exists.'
else:
	print 'Data file', fname,'does not exist. Try again.'
	quit()

file = open(fname,'r')
Data = np.loadtxt(file)

# FISTA Data
Time_FISTA = np.transpose(Data[1][:])
objective_FISTA = np.transpose(Data[0][:])
objective_decrease_FISTA = np.zeros((int(Iterations)-1,1))
objective_decrease_FISTA = objective_FISTA[0] - objective_FISTA

# OSTR Data
Time_OSTR = np.transpose(Data[3][:])
objective_OSTR = np.transpose(Data[2][:])
objective_decrease_OSTR = np.zeros((int(Iterations)-1,1))
objective_decrease_OSTR = objective_OSTR[0] - objective_OSTR 
	

# Plot the objective function vs time
pp.figure(1)
pp.plot(Time_FISTA, objective_FISTA, label='FISTA')
pp.plot(Time_OSTR, objective_OSTR, label='OSTR')
pp.title('Plot of objective function vs time')
pp.ylabel('objective function')
pp.xlabel('Time (seconds)')
pp.legend()
pp.show()
pp.savefig('Figures/Objective_Function_vs_Time_' + Iterations + '_Iterations.png')
pp.close()
# Plot the objective function decrease vs time
pp.figure(2)
pp.plot(Time_FISTA,objective_decrease_FISTA, label='FISTA')
pp.plot(Time_OSTR,objective_decrease_OSTR, label='OSTR')
pp.title('Plot of objective function decrease vs time')
pp.ylabel('objective function decrease')
pp.xlabel('Time (seconds)')
pp.show()
pp.savefig('Figures/Objective_Function_Decrease_vs_Time_' + Iterations + '_Iterations.png')
pp.close()
