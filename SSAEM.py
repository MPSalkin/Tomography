#SSAEM - Ordered Subsets Transmission
import pyraft as pr
import matplotlib.pyplot as pp
# The discrete randon transform method is imported first here
import fourier_slice as fs
import pseudopolar_fourier_slice as ppfs
# other methods must import from their respective files/libraries
import numpy as np
import math
import time
import savedata as sd
from skimage.measure import compare_ssim as ssim
import P_INTERP_PP as PIPP

# Function used to import data and radon function for data for nfft
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
    function = fs.make_fourier_slice_radon_transp
    fast_radon, fast_transp = function( sino )
    
    return sl, sino, counts, dark, flat, fast_radon, fast_transp, function      

# Function used to import data and radon function for data for nfft   
def pseudoimportSino(data_case):
    #You can custom import data by loading into directory and changing string
    #Sinogram must by NxN wher N is doubly even. It may be useful to develop 
    #interpolation function in feature domain to create artificial oversampling
    #this way any data can be made NxN with N divisible by 4.
    if data_case ==0:
        sl = pr.image_read( 'TomoData/PhantomData/sl2.mat', dtype=np.float32) 
        flat = pr.image_read( 'TomoData/PhantomData/slflat.mat', dtype=np.float32  ) 
        dark = pr.image_read( 'TomoData/PhantomData/sldark.mat', dtype=np.float32  )
        counts = pr.image_read( 'TomoData/PhantomData/slcount.mat', dtype=np.float32 ) 
    elif data_case == 1:
        sl = pr.image_read( 'TomoData/PhantomData/sl2.mat', dtype=np.float32) 
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
    #the lines below must always be performed regardless of data
    n,k = sl.shape
    #Form sino for interpolation step
    sino = np.log(flat/counts)
    #Pad Sino to eliminate need for extrapolation, oversample
    sino = PIPP.Pad2x(sino)
    print(np.max(sino))
    #Convert to frequency domain
    fsino = np.fft.fftshift(sino, axes = 0)
    fsino = np.fft.fft( fsino, axis = 0 )
    fsino = np.fft.fftshift(fsino)
    #Interpolate via 2 stage interpolation (reverse of averbuch et. al)
    fsino = PIPP.P_INTERP_PP(fsino)
    #Convert back to feature domain
    fsino = np.fft.fftshift(fsino )
    fsino = np.fft.ifft( fsino, axis = 0 )
    sino= (np.real(np.fft.ifftshift(fsino, axes=0)))
    sino= pr.image(np.concatenate((np.fliplr(np.flipud(sino[:,0:n])),np.fliplr(sino[:,n::])),1), top_left = (0,1), bottom_right = (np.pi,-1) )
    print(np.min(counts))
    #Reparse into counts,flat,dark
    counts = pr.image(flat/np.exp(sino), top_left = (0,1), bottom_right = (np.pi,-1) )
    function = ppfs.make_fourier_slice_radon_transp
    fast_radon, fast_transp = function( sino )

    return sl, sino, counts, dark, flat, fast_radon, fast_transp, function     

    
# Gradient function (y = counts; b = flat; r = dark)
def grad(l,counts,dark,flat):
    const = flat * np.exp(-l)
    tmp = ((counts)/(dark+const)-1)*const 
    return tmp

# Objective function
def hreg(l,counts,dark,flat):
    tmp = flat*np.exp(-l) + dark
    tmp1 = tmp
    tmp[tmp<0]=0
    tmp=np.log(tmp)
    tmp[tmp==1]=0
    tmp = tmp1 - counts*tmp
    return tmp

# Split image into M subests 
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

def ImportData(method_import,data_case):
    #This selects import data and gives a lambda paramter based on inputs
    if method_import == 0:
        A, b, counts, dark, flat, fast_radon, fast_transp, function = nonuimportSino(data_case)
        N = [4.05205*10**(-1),4.25*10**(-1),5.0*10**(-1)] #stored lambda values # 4 subsets
        #N = [5.0*10**(1),3.5*10**(1),2.15*10**(1)] #stored lambda values # X subsets
        return A, counts, dark,flat, fast_radon, fast_transp, function, N[data_case]
    elif method_import == 1:
        A, b, counts, dark, flat, fast_radon, fast_transp, function = pseudoimportSino(data_case)
        N = [5.0*10**(0),4.56365*10**(0) ,4.0*10**(0)] #stored lambda values
        return A, counts, dark,flat, fast_radon, fast_transp, function, N[data_case]
    elif method_import == 2:
        A, b, counts, dark, flat, fast_radon, fast_transp, function  = polarimportSino(data_case)
        N = [1.1*10**(-4),1.1*10**(-4),1.1*10**(-4)] #stored lambda values 
        return A, counts, dark,flat, fast_radon, fast_transp, function, N[data_case]
    else:
        print('choose a valid import case')
        quit()

if __name__ == "__main__":
#******************************************************************************    
#This is where you specify the method you're using and the data set. Custom
#data sets can be used by specifying new import locations in  the ""importSino()
#function for each respective method. Parameters are set in the N vector in the
#ImportData() function. N is length 3 (one for each data case) and you should
#be careful to adjust the corresponding lambda paramter to your method and data
#case.

#Warning! currently PPFFT allows for only 2 subsets and PFFT is not supported.

    Method = 1 #0 = NFFT, 1 = PPFFT, 2 = PFFT
    Data = 2  #0 = CLEAN PHANTOM, 1 = NOISY PHANTOM, 2=APPLESEED 
    M = 2 # Number of subsets
    N = 2 # Number of iterations
#******************************************************************************   
    # Import Ordered Subset data
    IMAGE, tmp_counts, tmp_dark, tmp_flat, ffast_radon, ffast_transp, FUN, lam = ImportData(Method,Data)
    # Get dimensions of sinogram
    row,col = tmp_counts.shape
    row2,col2 = IMAGE.shape
    # initialize splits and image
    x = np.ones((row2,row2))*0.9
    #x = np.ones((row2,ro2))*0.1 #adjusting starting guess matters!!!
    #x = np.ones((row2,row2))*(np.sum(tmp_counts) / np.sum( ffast_radon( np.ones((row2,row2)) ) ) )

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
        fast_radon[i], fast_transp[i] = FUN( sino )
        counts[i] = tmp_counts[:,col_M[i]]
        dark[i] = tmp_dark[:,col_M[i]]
        flat[i] = tmp_flat[:,col_M[i]]
  
    # dj = fast_radon(gamma*c(l))
    T = np.zeros((N,1))
    obj = np.zeros((N,1))
    SSIM = np.zeros((N,1))
    subseth = np.zeros((M,1))
    itr = 0
    row,col = x.shape
    subiter_time = np.zeros((M,1))
    D = np.zeros((row,col))
    # Preallocate parameters
    #lam = 1.5*10**(1)#5.0*10**(1) #3.5*10**(1)#2.15*10**(1) #for noisy data #3.0625*10**(0)
    #lam = 3.7*1-1**(1)#5.0*10**(1) #3.7*10**(1)#3.12*10**(1)# #for clean data
    #lam = 4.56365*10**(0) # pseudopolar noisy
    #lam = 5.0*10**(0) # pseudopolar clean
    #lam = 4.0*10**(0) # pseudopolar appleseed
    tau = 10**(-8)#1.1*10**(-4)
    lam0 = lam
    #pj = ffast_transp(tmp_counts-tmp_dark)
    pj = ffast_transp(tmp_flat*np.exp(-ffast_radon(x)))
    # Main loop for SSAEM image reconstruction algorithm
    start_time = time.time()
    for n in range(0,N):
        iter_begin = time.time()
        print 'Iteration:',itr+1
        
        # Nested loop to iterate over M subsets
        for mm in np.random.permutation(M):
            #subiter_start = time.time()
            g = fast_transp[mm](grad(fast_radon[mm](x),counts[mm],dark[mm],flat[mm]))
            D = x
            D[(x<=tau) & (g<=0)] = tau
            D = D/pj
            x = x - lam*D*g
            print(np.min(x))
        lam = lam0/(n+1)**0.25
        
            #subiter_time[mm] = time.time()-subiter_start
            #subseth[mm] = np.sum(hreg(l,counts[mm],dark[mm],flat[mm]))
            
        #Store time immediately after iteration
        #iteration_time = np.sum(subiter_time)
        iteration_time = time.time() - iter_begin
        if n==0 :
            T[n,0] = iteration_time - start_time + iter_begin
        else:
            T[n,0] = iteration_time + T[n-1,0]
        SSIM[n,0] = ssim(IMAGE,x/np.max(x))
        #Compute and store objective function
        obj[n,0] = np.sum(hreg(ffast_radon(x),tmp_counts,tmp_dark,tmp_flat))
        
        itr += 1
           
            

    # Save objective function and time values
    sd.saveme(obj,T,SSIM,N,M)

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
    SSAEM_image=pp.imshow( image, cmap = 'gray', interpolation = 'nearest', 
              extent = ( image.top_left[ 0 ], image.bottom_right[ 0 ], image.bottom_right[ 1 ], image.top_left[ 1 ] ))
    pp.title('SSAEM ' + str(M) + ' Subsets - ' + str(N) + 'Iterations')
    pp.savefig('Visuals/Images/SSAEM_noisy_reconstruct_'+ str(N) + '_Iter_' + str(M) + '_Subsets.png', format='png', dpi=200)
    pp.colorbar(SSAEM_image)
    pp.show(SSAEM_image)

