#FISTA - Fast Iterative Shrinkage-Thresholding Algorithm
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math
import time 
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


def importPhoto():
	sl = pr.image_read( 'sl.mat' ) 

	#pp.imshow( sl, cmap = 'gray_r', interpolation = 'nearest', 
#		extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
	#pp.show()

	sino = pr.image( np.zeros( (1024,1024) ) , top_left =  (0,1), bottom_right = (math.pi, -1) ) 

	fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )

	sino = fast_radon( sl ) 

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
	#pp.show()
	
	return sl, sino, fast_radon, fast_transp

def importSino():
	sino = pr.image_read( 'egg_slice_1097.mat' ) 

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
	#pp.show()

	fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
	#pp.show()
	
	return sino, fast_radon, fast_transp

def grad(x):
	tmp = fast_radon(x) - b
	return (tmp)

def ImportData(sino_or_image):
	if sino_or_image == 0:
		A,b, fast_radon, fast_transp = importPhoto()
	elif sino_or_image == 1:
		b, fast_radon, fast_transp = importSino()
		A = np.zeros((np.min(b.shape),np.min(b.shape)))
  
	return A,b, fast_radon, fast_transp

#def FISTA(sino_or_image)
RealData = 0
if RealData == 0:
   A, b, fast_radon, fast_transp = ImportData(0)
elif RealData == 1:
   A, b, fast_radon, fast_transp = ImportData(1)
#img = img_as_float(A)
BEST = np.zeros(A.shape)
m,n = b.shape

gamma = 0.45
tau = pow(10,-4)
y = np.ones((m,m)) * ( np.sum(b) / np.sum( fast_radon( np.ones((m,m)) ) ) )
x0 = np.ones((m,m))
t = 1
c = 0
x = y 

# while np.linalg.norm(x - x0, 'fro')/np.linalg.norm(x0, 'fro') > tau:
# 	if c > 12:
# 		break
itr = 50
T = np.zeros((itr,1))
obj = np.zeros((itr,1))
SSIM = np.zeros((itr,1))
start_time = time.time()
for i in range(0,itr):
	x0 = x
	t0 = t

	c += 1
	print(c)
	x = y - gamma*fast_transp(grad(x0))
	x[x<0] = 0
	t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
	y = x + (t0 - 1) / t * (x - x0)

	# Record time of each iteration
	T[i,0] = time.time() - start_time

	# Store the objective function for each iteration.
	obj[i,0] = np.linalg.norm(grad(x0))**2
	# Store the image and SSIM for each iteration.
	SSIM[i,0] = ssim(A,x)
        


print(T)
print(obj)
obj2 = np.zeros((itr-1,1))
obj2 = obj[0,0] - obj
#obj = np.log10(obj)
#print(x)
#pp.figure(1)
#image = pr.image( x , top_left =  (-1,1), bottom_right = (1, -1) ) 
#pp.imshow( image, cmap = 'gray_r', interpolation = 'nearest', 
	#	extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ]#, image.bottom_right[ 1 ], image.top_left[ 1 ] ))
#pp.show()

pp.figure(1)
pp.plot(T,obj)
pp.ylabel('objective function')
pp.xlabel('Time (in seconds)')
pp.show()

pp.figure(2)
pp.plot(T,obj2)
pp.ylabel('objective function decrease')
pp.xlabel('Time (in seconds)')
pp.show()

pp.figure(3)
pp.plot(T,SSIM)
pp.ylabel('SSIM')
pp.xlabel('Time (in seconds)')
pp.show()
#pp.xlabel('Time')
#pp.ylabel('Objective Function')