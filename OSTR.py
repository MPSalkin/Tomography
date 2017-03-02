#OSTR - Ordered Subsets Transmission
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math
import time
import savedata as sd
from skimage.measure import compare_ssim as ssim

# Function used to import data and radon function for data
def importOS():
    IMAGE = pr.image_read( 'TomoData/PhantomData/sl.mat')
    counts = pr.image_read( 'TomoData/noisyphantom/nslcounts.mat', dtype=np.float32 ) 
    print 'counts',counts.shape
    dark = pr.image_read( 'TomoData/noisyphantom/nsldark.mat', dtype=np.float32  ) 
    print 'dark', dark.shape
    flat = pr.image_read( 'TomoData/noisyphantom/nslflat.mat', dtype=np.float32  ) 
    print 'flat', flat.shape
    # flat = np.zeros(counts.shape)
    #pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
		#extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
    #pp.show()

    fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( counts )

    #pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
               #extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ))
    #pp.show()
	
    return IMAGE, counts, dark, flat, fast_radon, fast_transp

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
    IMAGE, tmp_counts, tmp_dark, tmp_flat, ffast_radon, ffast_transp = importOS()
    
    # Get dimensions of sinogram
    row,col = tmp_counts.shape

    # initialize splits and image
    M = 10 # Number of subsets
    N = 20 # Number of iterations
    x = np.zeros((row,row)) * ( np.sum(tmp_counts) / np.sum( ffast_radon( np.ones((row,row)) ) ) )
    
    # Create M subsets of sinogram
    col_M = list(split(range(col),M))
    fast_radon = [None]*M
    fast_transp = [None]*M
    counts = [None]*M
    dark = [None]*M
    flat = [None]*M
    
    # Create fast radon/transp functions and subdivide data for subsets
    for i in range(0,M): 
        len_col_m = len(col_M[i])
        sino = pr.image( np.zeros( (row, len_col_m) ) , 
                        top_left =  (col_M[i][0]*math.pi/(col-1), 1), bottom_right = (col_M[i][-1]*math.pi/(col-1), -1) ) 
        tmp_radon, tmp_transp = fs.make_fourier_slice_radon_transp( sino )
        fast_radon[i] = tmp_radon
        fast_transp[i] = tmp_transp
        counts[i] = tmp_counts[:,col_M[i]]
        dark[i] = tmp_dark[:,col_M[i]]
        flat[i] = tmp_flat[:,col_M[i]]
  
    # dj = fast_radon(gamma*c(l))
    T = np.zeros((N,1))
    obj = np.zeros((N,1))
    SSIM = np.zeros((N,1))
    subseth = np.zeros((M,1))
    itr = 0
    subiter_time = np.zeros((M,1))
    start_time = time.time()
    # Preallocate parameters
    gamma = ffast_radon( np.ones((row,row)) )
    d_star = ffast_transp( gamma * (tmp_counts - tmp_dark)**2 / tmp_counts )
    # Main loop for OSTR image reconstruction algorithm
    for n in range(0,N):
        iter_begin = time.time()
        # Nested loop to iterate over M subsets
        for mm in np.random.permutation(M):
            #subiter_start = time.time()
            l = fast_radon[mm](x)
            h_dot = grad(l,counts[mm],dark[mm],flat[mm])
            L_dot = fast_transp[mm](h_dot)     
            x = x - M*L_dot/d_star
            #subiter_time[mm] = time.time()-subiter_start
            #subseth[mm] = np.sum(hreg(l,counts[mm],dark[mm],flat[mm]))
            
        #Store time immediately after iteration
        #iteration_time = np.sum(subiter_time)
        iteration_time = time.time() - iter_begin
        if n==0 :
            T[n,0] = iteration_time - start_time + iter_begin
        else:
            T[n,0] = iteration_time + T[n-1,0]
        itr += 1
        print 'Iteration:',itr
        SSIM[n,0] = ssim(IMAGE,x)
        #compute elapsed time
        #Compute and store objective function
        obj[n,0] = np.sum(hreg(ffast_radon(x),tmp_counts,tmp_dark,tmp_flat))
        #Keep track of itreations
           
            

    # Save objective function and time values
    sd.saveme(obj,T,SSIM,N,M,'OSTR')

    #Display Time and Objective function vectors.
#    print(T)
#    print(obj)
#    print('end outputs')
    
    #Compute objective function decrease
    obj2 = np.zeros((itr-1,1))
    obj2 = obj[0,0] - obj

    # Print Recovered Image, Time, and Objective Function data
    print 'ImageData', x
    print 'Time:', T
    print 'ObjectiveFunction', obj
    # sino_x = ffast_radon(x)
    # image_sino = pr.image( sino_x , top_left =  (0,1), bottom_right = (math.pi, -1) ) 
    # pp.imshow( image_sino, cmap = 'gray_r', interpolation = 'nearest', 
    # 		extent = ( image_sino.top_left[ 0 ], image_sino.bottom_right[ 0 ], image_sino.bottom_right[ 1 ], image_sino.top_left[ 1 ] ))
    # pp.show()

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
    
    # Display reconstructed image
    pp.figure(4)
    image = pr.image( x , top_left =  (-1,1), bottom_right = (1, -1) ) 
    pp.imshow( image, cmap = 'gray', interpolation = 'nearest', 
              extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
    pp.title('OSTR ' + str(M) + ' Subsets - ' + str(N) + 'Iterations (Moderate Noise)')
    pp.savefig('Visuals/Images/OSTR_noisy_reconstruct_'+ str(N) + '_Iter_' + str(M) + '_Subsets.png')
    pp.show()

