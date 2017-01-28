import pyraft as pr
import matplotlib.pyplot as pp
import fourier_slice as fs
import numpy as np
import math

# From Fullerton_Example 
# implementation of FISTA 
# uses actual data (seed data)

yy=pr.image_read('counts.mat',dtype=np.float32)  
r=pr.image_read('dark.mat',dtype=np.float32)  
b=pr.image_read('flat.mat',dtype=np.float32)  

dim = 2048

       # USE THIS TO CHANGE DIMENSIONS OF DATA 
sino = yy # given sinogram data 
fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )
#pp.imshow(sino, interpolation = 'nearest', cmap = 'gray_r', extent = ( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.bottom_right[ 1 ], sino.top_left[ 1 ] ) )
#pp.show()

def gradient(xz):
    temp = (b*yy)/(b+np.exp(fast_radon(xz))*r) - b*np.exp(-fast_radon(xz))
    return fast_transp(temp)



told = 1 # initial t val
data1 = sino # locate image
gamma = 0.6 # flexible parameter, gamma
sc = np.ones((dim,1)) 
alpha = np.sum(data1)/np.sum(data1*sc)
y = np.ones((dim,dim))
y = alpha*y
xold = y
xnew = 100*y; 
toler = 0.0001  
iterz=0 # initiate iteration count 

while (np.linalg.norm(xnew-xold)/np.linalg.norm(xold)>toler):
    xold = xnew
    xnew = y - gamma*gradient(xold)
    xnew[xnew<0]=0 
    cc = np.power(told,2)
    cd = np.multiply(4,cc)
    ce = np.sqrt( 1 + cd)
    cf = 1 + ce
    cg = np.divide(cf,2)
    tnew = cg
    y = xnew + np.multiply(np.divide(told - 1,tnew),(xnew - xold))
     
    if iterz > 10:
        break 
    
    iterz = iterz+1
    print iterz
    told = tnew
   
Image = pr.image(xnew,top_left=(-1,1),bottom_right=(1,-1))
pp.imshow(Image, interpolation = 'nearest', cmap = 'gray_r', extent = ( Image.top_left[ 0 ], Image.bottom_right[ 0 ], Image.bottom_right[ 1 ], Image.top_left[ 1 ] ) )
pp.show()
 
    #print(y)

#told = tnew
#xold = xnewddssm
    
    

