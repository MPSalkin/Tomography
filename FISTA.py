#FISTA - Fast Iterative Shrinkage-Thresholding Algorithm
import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import pseudopolar_fourier_slice as ppfs
import numpy as np
import math
import time 
import savedata as sd
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
import P_INTERP_PP as PIPP

# Function used to import sinogram data and radon function for data
def nonuimportSino(data_case):
    if data_case ==0:
        sl = pr.image_read( 'TomoData/PhantomData/sl.mat') 
        flat = pr.image_read( 'TomoData/PhantomData/slflat.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/PhantomData/sldark.mat', dtype=np.float32  )
        counts = pr.image_read( 'TomoData/PhantomData/slcount.mat', dtype=np.float32 ) 
    elif data_case ==1:
        sl = pr.image_read( 'TomoData/PhantomData/sl.mat') 
        counts = pr.image_read( 'TomoData/noisyphantom/nslcounts.mat', dtype=np.float32 ) 
        flat = pr.image_read( 'TomoData/noisyphantom/nslflat.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/noisyphantom/nsldark.mat', dtype=np.float32  )
    elif data_case ==2:
        sl = pr.image(np.ones((2048,2048)), top_left =(-1,1), bottom_right = (1,-1)) 
        flat = pr.image_read( 'TomoData/SeedData/flati.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/SeedData/darki.mat', dtype=np.float32  ) 
        counts = pr.image_read( 'TomoData/SeedData/countsi.mat', dtype=np.float32 ) 
    else:
        print('not a valid data_case, you can insert custom data here')
        quit()
 

    sino = pr.image(np.log(flat/counts), top_left = (0,1),bottom_right=(np.pi,-1))
    fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )
    
    return sl, sino, counts, dark, flat, fast_radon, fast_transp
     
def pseudoimportSino(data_case):
    if data_case ==0:
        sl = pr.image_read( 'TomoData/PhantomData/sl2.mat', dtype=np.float32) 
        flat = pr.image_read( 'TomoData/PhantomData/slflat.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/PhantomData/sldark.mat', dtype=np.float32  )
        counts = pr.image_read( 'TomoData/PhantomData/slcount.mat', dtype=np.float32 ) 
    elif data_case == 1:
        flat = pr.image_read( 'TomoData/noisyphantom/nslflat.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/noisyphantom/nsldark.mat', dtype=np.float32  ) 
        counts = pr.image_read( 'TomoData/noisyphantom/nslcounts.mat', dtype=np.float32 ) 
    elif data_case ==2:
        sl = pr.image(np.ones((1024,1024)), top_left =(-1,1), bottom_right = (1,-1)) 
        flat = pr.image_read( 'TomoData/SeedData/flati.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/SeedData/darki.mat', dtype=np.float32  ) 
        counts = pr.image_read( 'TomoData/SeedData/countsi.mat', dtype=np.float32 ) 
    else:
        print('not a valid data_case, you can insert custom data here')
        quit()
    n,k = sl.shape
    sino = np.log(flat/counts)
    sino = PIPP.Pad2x(sino)
    print(np.max(sino))
    fsino = np.fft.fftshift(sino, axes = 0)
    fsino = np.fft.fft( fsino, axis = 0 )
    fsino = np.fft.fftshift(fsino)
    fsino = PIPP.P_INTERP_PP(fsino)
    fsino = np.fft.fftshift(fsino )
    fsino = np.fft.ifft( fsino, axis = 0 )
    sino= (np.real(np.fft.ifftshift(fsino, axes=0)))
    sino= pr.image(np.concatenate((np.fliplr(np.flipud(sino[:,0:n])),np.fliplr(sino[:,n::])),1), top_left = (0,1), bottom_right = (np.pi,-1) )
    print(np.min(counts))
    counts = pr.image(flat/np.exp(sino), top_left = (0,1), bottom_right = (np.pi,-1) )
    
    fast_radon, fast_transp = ppfs.make_fourier_slice_radon_transp( sino )
    
    return sl, sino, counts, dark, flat, fast_radon, fast_transp     


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

def ImportData(method_import,data_case):
    if method_import == 0:
        A, b, counts, dark, flat, fast_radon, fast_transp = nonuimportSino(data_case)
        N = [1.1*10**(-4),8.5*10**(-5),5.0*10**(-5)] #stored gamma values 
        return A, b, counts, dark,flat, fast_radon, fast_transp, N[data_case]
    elif method_import == 1:
        A, b, counts, dark, flat, fast_radon, fast_transp = pseudoimportSino(data_case)
        N = [5.4*10**(-1),2.7*10**(-4),7.2*10**(-5)] #stored gamma values 
        return A, b, counts, dark,flat, fast_radon, fast_transp, N[data_case]
    elif method_import == 2:
        A, b, counts, dark, flat, fast_radon, fast_transp = polarimportSino(data_case)
        N = [1.1*10**(-4),1.1*10**(-4),1.1*10**(-4)] #stored gamma values 
        return A, b, counts, dark,flat, fast_radon, fast_transp, N[data_case]
    else:
        print('choose a valid import case')
        quit()
       

Method = 0 #0 = NFFT, 1 = PPFFT, 2 = PFFT
Data = 2  #0 = CLEAN PHANTOM, 1 = NOISY PHANTOM, 2=APPLESEED
itr = 5

# Data Import Functions
A, b, counts, dark, flat, fast_radon, fast_transp, gamma = ImportData(Method,Data)
BEST = np.zeros(A.shape)
m,n = A.shape

print m,n
print 'Image', A.shape
print(np.min(counts))
print(np.min(A))

# variables that can be precomputed with knowledge of geometry of problem
T = np.zeros((itr,1))
obj = np.zeros((itr,1))
SSIM = np.zeros((itr,1))

# Constants and initialization

tau = pow(10,-4)
y = np.ones((m,m)) * ( np.sum(b) / np.sum( fast_radon( np.ones((m,m)) ) ) )
x0 = np.ones((m,m))
t = 1
count = 0
x = y 
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
    SSIM[i,0] = ssim(A,x/np.max(x))
    if i==0 :
        T[i,0] = current_time - start_time
    else:
        T[i,0] = (current_time - iter_begin) + T[i-1,0]
          
    # Store the objective function for each iteration.
    obj[i,0] = np.sum(hreg(fast_radon(x),counts,dark,flat))
    
    # Store the image and SSIM for each iteration.
    #SSIM[i,0] = ssim(A,x)
        
# Save objective function and time values
sd.saveme(obj,T,SSIM,itr,0)

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
image = pr.image((x) , top_left =  (-1,1), bottom_right = (1, -1) ) 
FISTA_image = pp.imshow( image,cmap = 'gray', interpolation = 'nearest', 
		extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
pp.title('FISTA (ML) - 10 Iterations')
pp.savefig('Visuals/Images/FISTA_noisy_reconstruct_' + str(itr) + '_ITR' '.png', format='png', dpi=200)
pp.colorbar(FISTA_image)
pp.show(FISTA_image)


#pp.figure(4)
#FISTA_image = pp.imshow( image, cmap = 'gray', interpolation = 'nearest', 
#		extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
#pp.title('Shepp Logan Phantom - 1024x1024')
#pp.savefig('Phantom.png')