import numpy as np
import pyraft as pr
import math

# F = open('data.txt','w')

# a = np.array() #[10,9,8.7,6,5,4,3,2,1]

# F.write(a)
# F.close()

# print(a)

# print(F)



a = [1,2,3,4,5]
b = [2,6,4,3,2]
c = [2,4,6,8,2]
A =  [[1,2,3,4,5],[2,6,4,3,2],[2,4,6,8,2]]
A = np.zeros((4,10))
#i,j = A.shape
print('A',A)

# f = open('file.txt', 'wb')
# for i in range(len(a)):
    # f.write("%i %5.2f\n" % (a[i], b[i]))

np.savetxt('Objective_Function_data_5itr.txt', A)
f = open('data.txt','r')
m = np.loadtxt(f, usecols=(0,2), unpack=True)
print(m)

#, y, z = np.loadtxt('file_numpy.txt')
#print(x,y,z)
#A[1][:] = [1,2,3,4,0]

np.savetxt('data.txt', A)
f = open('data.txt','r')
m = np.loadtxt(f)
print('m',m)

f.close()

