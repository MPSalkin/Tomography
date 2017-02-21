#FISTA - Fast Iterative Shrinkage-Thresholding Algorithm
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math
import time 
import savedata as sd
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim

# Function used to import photo data and radon function for data
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

# Function used to import sinogram data and radon function for data
def importSino():
	# sino = pr.image_read( 'egg_slice_1097.mat' ) 
	sl = pr.image_read( 'sl.mat' ) 
	# sino = pr.image_read( 'slcount.mat', dtype=np.float32 ) 
	# flat = pr.image_read( 'slflat.mat', dtype=np.float32  ) 
	# dark = pr.image_read( 'sldark.mat', dtype=np.float32  )
	counts = pr.image_read( 'TomoData/noisyphantom/nslcounts.mat', dtype=np.float32 ) 
	flat = pr.image_read( 'TomoData/noisyphantom/nslflat.mat', dtype=np.float32  ) 
	dark = pr.image_read( 'TomoData/noisyphantom/nsldark.mat', dtype=np.float32  ) 
	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
	#pp.show()
	sino = np.log(flat/counts)

	fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )

	#pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
	#pp.show()
	return sl, sino, counts, dark, flat, fast_radon, fast_transp

# # Gradient function
# def grad(x):
# 	tmp = fast_radon(x) - b
# 	return (tmp)
# Gradient function (y = counts; b = flat; r = dark)
def grad(l,counts,dark,flat):
    const = flat * np.exp(-l)
    tmp = ((counts)/(dark+const)-1)*const 
    return tmp

# Objective function
def hreg(l,counts,dark,flat):
    tmp1 = tmp = flat*np.exp(-l) + dark
    tmp[tmp<0]=0
    tmp=np.log(tmp)
    tmp[tmp==1]=0
    tmp = tmp1 - counts*tmp
    return tmp

# Function to select type of data to import
def ImportData(sino_or_image):
	if sino_or_image == 0:
		A,b, fast_radon, fast_transp = importPhoto()
		return A,b, fast_radon, fast_transp
	elif sino_or_image == 1:
		A, b, counts, dark, flat, fast_radon, fast_transp = importSino()
		# A = np.zeros((np.min(b.shape),np.min(b.shape)))
		return A, b, counts, dark,flat, fast_radon, fast_transp
  

RealData = 1
if RealData == 0:
   A, b, fast_radon, fast_transp = ImportData(0)
elif RealData == 1:
   A, b, counts, dark, flat, fast_radon, fast_transp = ImportData(1)
#img = img_as_float(A)
BEST = np.zeros(A.shape)
m,n = b.shape

print m,n
print 'Image', A.shape

# Constants and initialization
gamma = 1.1*10**(-4)
tau = pow(10,-4)
y = np.ones((m,m)) * ( np.sum(b) / np.sum( fast_radon( np.ones((m,m)) ) ) )
x0 = np.ones((m,m))
t = 1
count = 0
x = y 

itr = 20
T = np.zeros((itr,1))
obj = np.zeros((itr,1))
SSIM = np.zeros((itr,1))
start_time = time.time()

# Main loop for FISTA image reconstruction algorithm
for i in range(0,itr):
	# Print iteration number
    count += 1
    print 'Iteration:',count

    #Begin clock at each iteration
    iter_begin = time.time()

    # FISTA meat and potatos
    x0 = x   
    t0 = t
    x = y - gamma*fast_transp(grad(fast_radon(x0),counts,dark,flat))
    x[x<0] = 0
    t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
    y = x + (t0 - 1) / t * (x - x0)

    # Record time of each iteration
    current_time = time.time()
    SSIM[i,0] = ssim(A,x)
    if i==0 :
        T[i,0] = current_time - start_time
    else:
        T[i,0] = (current_time - iter_begin) + T[i-1,0]
          
    # Store the objective function for each iteration.
    obj[i,0] = np.sum(hreg(fast_radon(x),counts,dark,flat))
    
    # Store the image and SSIM for each iteration.
    #SSIM[i,0] = ssim(A,x)
        
# Save objective function and time values
sd.saveme(obj,T,SSIM,itr,0,'FISTA')

# Print Recovered Image, Time, and Objective Function data
print 'ImageData', x
print 'Time:', T
print 'ObjectiveFunction', obj
#obj2 = np.zeros((itr-1,1))
obj2 = obj[0,0] - obj
#obj = np.log10(obj)
#print(x)

# Plot the objective function vs time
pp.figure(1)
pp.plot(T,obj)
pp.ylabel('objective function')
pp.xlabel('Time (in seconds)')
pp.show()

# Plot the objective function decrease vs time
pp.figure(2)
pp.plot(T,obj2)
pp.ylabel('objective function decrease')
pp.xlabel('Time (in seconds)')
pp.show()

# Plot the SSIM 
# pp.figure(3)
# pp.plot(SSIM)
# pp.ylabel('SSIM')
# pp.xlabel('Iteration')
# pp.show()
#pp.xlabel('Time')
#pp.ylabel('Objective Function')

# Display reconstructed image
pp.figure(4)
image = pr.image( x , top_left =  (-1,1), bottom_right = (1, -1) ) 
FISTA_image = pp.imshow( image, cmap = 'gray', interpolation = 'nearest', 
		extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
pp.title('FISTA (ML) - 20 Iterations (Moderate Noise)')
pp.savefig('Visuals/Images/FISTA_noisy_reconstruct_' + str(itr) + '_ITR' '.png')
pp.show(FISTA_image)
