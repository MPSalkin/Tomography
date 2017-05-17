#OSEM - Ordered Subset Expectation Maximization 
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math

def importPhoto():
	sl = pr.image_read( 'sl.mat' ) 

	#pp.imshow( sl, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
	#pp.show()

	sino = pr.image( np.zeros( (1024,1024) ) , top_left =  (0,1), bottom_right = (math.pi, -1) ) 

	fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )

	sino = fast_radon( sl ) 

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
	#pp.show()
	
	return sino, fast_radon, fast_transp

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

def importOSTR():
	counts = pr.image_read( 'counts.mat', dtype=np.float32 ) 
	dark = pr.image_read( 'dark.mat', dtype=np.float32  ) 
	flat = pr.image_read( 'flat.mat', dtype=np.float32  ) 

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
	#pp.show()

	fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( counts )

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
	#pp.show()
	
	return counts, dark, flat, fast_radon, fast_transp

# def grad(x):
# 	tmp = fast_radon(x) - b
# 	return fast_transp(tmp)

def grad(x):
	tmp = (flat*counts) / ( flat + dark * np.exp(fast_radon(x)) ) - flat * np.exp(-fast_radon(x))
	return fast_transp(tmp)

# def ImportData(dataSel):
# 	if dataSel == 0:
# 		b, fast_radon, fast_transp = importPhoto()
# 	elif dataSel == 1:
# 		b, fast_radon, fast_transp = importSino()
# 	elif dataSel == 2:
# 		counts, dark, flat, fast_radon, fast_transp = importOSTR()

	# return b, fast_radon, fast_transp

#def FISTA(sino_or_image)
b, dark, flat, fast_radon, fast_transp = importOSTR()

counts = b
m,n = b.shape
print m,n

gamma = 10**(-4)
tau = pow(10,-20)
y = np.zeros((m,m)) * ( np.sum(b) / np.sum( fast_radon( np.ones((m,m)) ) ) )
x0 = np.ones((m,m))
t = 1
c = 0
x = y 

## Main loop for reconstruction algorithm
while np.linalg.norm(x - x0, 'fro')/np.linalg.norm(x0, 'fro') > tau:
	if c > 5:
		break
	x0 = x
	t0 = t
	
	x = y - gamma*grad(x0)
	x[x<0] = 0
	t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
	y = x + (t0 - 1) / t * (x - x0)

	c += 1
	print(c)

## Print Recovered Image
print(x)
sino_x = fast_radon(x)
image_sino = pr.image( sino_x , top_left =  (0,1), bottom_right = (math.pi, -1) ) 
pp.imshow( image_sino, cmap = 'gray_r', interpolation = 'nearest', 
		extent = ( image_sino.top_left[ 0 ], image_sino.bottom_right[ 0 ], image_sino.bottom_right[ 1 ], image_sino.top_left[ 1 ] ))
pp.show()
image = pr.image( x , top_left =  (-1,1), bottom_right = (1, -1) ) 
pp.imshow( image, cmap = 'gray_r', interpolation = 'nearest', 
		extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
pp.show()

