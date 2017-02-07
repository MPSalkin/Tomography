#OSTR - Ordered Subsets Transmission
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math
import time
from skimage.measure import compare_ssim as ssim

def importOS():
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

# Gradient function (y = counts; b = flat; r = dark)
def grad(l,counts,dark,flat,idx):
    tmp = (flat*counts) / ( flat + dark * np.exp(l) ) - flat * np.exp(-l)
    return tmp
    
def hreg(l,counts,dark,flat,idx):
    luis = flat*np.exp(-l) + dark
    luis[luis<0]=0
    luis=np.log(luis)
    luis[luis==1]=0
    tmp2 = flat*np.exp(-l) + dark - counts*luis
    return tmp2
# c(l) function
# def c_l(l,counts,dark,flat):
#     y_bar = flat * np.exp(-l) + dark	
#     l[l>0]= ( 2 / l ** 2) * (flat * (1 - np.exp(-l)) - counts * np.log((flat + dark) / y_bar) + l * flat * np.exp(-l) * (counts/y_bar -1))			
#     l[l==0] = flat * (1 - counts * dark / (flat + dark)**2 )
#     return l

# Split image into M subests 
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


if __name__ == "__main__":
    # Import Ordered Subset data
    tmp_counts, tmp_dark, tmp_flat, ffast_radon, ffast_transp = importOS()
    
    # Get dimensions of sinogram
    row,col = tmp_counts.shape
    # initialize splits and image
    M = 5 # Number of subsets
    N = 20 # Number of iterations
    x = np.zeros((row,row)) * ( np.sum(tmp_counts) / np.sum( ffast_radon( np.ones((row,row)) ) ) )
    
    # Create M subsets of sinogram
    col_M = list(split(range(col),M))
    
    fast_radon = []
    fast_transp = []
    counts = []
    dark = []
    flat = []
    
    # Create fast radon/transp functions for subsets
    
    for i in range(0,M): 
        len_col_m = len(col_M[i])
        sino = pr.image( np.zeros( (row, len_col_m) ) , 
                        top_left =  (col_M[i][0]*math.pi/(col-1), 1), bottom_right = (col_M[i][-1]*math.pi/(col-1), -1) ) 
        tmp_radon, tmp_transp = fs.make_fourier_slice_radon_transp( sino )
        fast_radon.append(tmp_radon)
        fast_transp.append(tmp_transp)
        counts.append(tmp_counts[:,col_M[i]])
        dark.append(tmp_dark[:,col_M[i]])
        flat.append(tmp_flat[:,col_M[i]])
    
    ## Main loop for reconstruction algorithm
    gamma = ffast_radon( np.ones((row,row)) )
    d_star = ffast_transp( gamma * (tmp_counts - tmp_dark)**2 / tmp_counts )
    itr = 0
    # dj = fast_radon(gamma*c(l))
    T = np.zeros((N,1))
    obj = np.zeros((N,1))
    SSIM = np.zeros((N,1))
    start_time = time.time()
    subseth = np.zeros((M,1))
    for n in range(0,N):
        iter_begin = time.time()
        for mm in range(0,M):
            l = fast_radon[mm](x)
            h_dot = grad(l,counts[mm],dark[mm],flat[mm],mm)
            L_dot = fast_transp[mm](h_dot)
            x = x - M*L_dot/d_star
            subseth[mm] = np.sum(hreg(l,counts[mm],dark[mm],flat[mm],mm))
        #Store time immediately after iteration
        current_time = time.time()
        #compute elapsed time
        if n==0 :
            T[n,0] = current_time - start_time
        else:
            T[n,0] = (current_time - iter_begin) + T[n-1,0]
        #Compute and store objective function
        obj[n,0] = np.sum(subseth)
        #Keep track of itreations
        itr += 1
        print(itr)
    #Display Time and Objective function vectors.
#    print(T)
#    print(obj)
#    print('end outputs')
    
    #Compute objective function decrease
    obj2 = np.zeros((itr-1,1))
    obj2 = obj[0,0] - obj
    ## Print Recovered Image
    print(x)
    # sino_x = ffast_radon(x)
    # image_sino = pr.image( sino_x , top_left =  (0,1), bottom_right = (math.pi, -1) ) 
    # pp.imshow( image_sino, cmap = 'gray_r', interpolation = 'nearest', 
    # 		extent = ( image_sino.top_left[ 0 ], image_sino.bottom_right[ 0 ], image_sino.bottom_right[ 1 ], image_sino.top_left[ 1 ] ))
    # pp.show()
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
    
    pp.figure(4)
    image = pr.image( x , top_left =  (-1,1), bottom_right = (1, -1) ) 
    pp.imshow( image, cmap = 'gray_r', interpolation = 'nearest', 
              extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
    pp.show()

