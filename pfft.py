#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 18:19:45 2017

@author: CupulFamily
"""
import pyraft as pr
import numpy as np
import math
import warnings
import matplotlib.pyplot as pp

def fft(img):
    
    A = np.ones((11,11))
    A = pr.image_read('sl69.mat', dtype=np.float32)
    
    #padding to nearest 2*(N+1)+1 <--- to make odd
    r1,c1 = A.shape
    global PAD, r
    PAD = (r1+3)/2
    print(PAD)
    A = np.pad(A, PAD, 'minimum')
    r,c = A.shape
    
    #Set various endpoints and midpoints
    N = r-1 #N, as used in II.5,II.6,II.9,II.10 etc.
    M = r1-1 #number of radial lines (not including 180degrees)
    m = M # assigning m for use in loop
    CTR = (N+1)/2
    ACTR = M/2
    dtheta = np.pi/M 
    
    
    
    if np.mod(M,4) == 0:
        L = M/4
    elif np.mod(M,2) == 0:
        L = (M-2)/4
     
    print(L)
    print('^^Required levels^^')   
    #initialization of FFT and FrFT matrices
    F = np.zeros((r,M))+0j
    fx = np.zeros((r,c))+0j
    fy = np.zeros((r,c))+0j       

    #map python indexes to -N/2 to N/2 cartesian grid
    C = range(-N/2,N/2 + 1) #column mapping
    R = C[::-1] #row mapping
    SN = C #n sum mapping for FrFt

    for l in range(1,L+1):
        # scaling by alpha_l
        alpha = np.cos(l*dtheta)
        
        #loop for II.5, II.6
        for i in range(0,r) :
            for j in range (0,c) :
                fx[i,j]= 0
                #n summation loop
                for n in range(0,r):
                    #II.5 FrFT along rows
                    fx[i,j] = fx[i,j] + A[i,n]*np.exp(-1j*2*np.pi*C[j]*alpha*SN[n]/(N+1))
       
        #conditional loop to accomodate doubly even angles
        if (m%4==0) & (l == L):
            pass #on the last level, skip extra calcs (only for doubly even M)
        else:
            for j in range (0,c) :  
                for i in range(0,r) :
                    fy[i,j] = 0
                    #n summation loop
                    for n in range(0,c):
                        #II.6 FrFT along columns
                        fy[i,j] = fy[i,j] + A[n,j]*np.exp(-1j*2*np.pi*R[i]*alpha*SN[n]/(N+1))
       
        #Scaling by Beta_l
        beta = np.sin(l*dtheta)
        
        #initialization of radial lines that will be overwritten and stored later
        Fx1 = np.zeros((1,r))+0j
        Fx2 = np.zeros((1,r))+0j
        Fy1 = np.zeros((1,r))+0j
        Fy2 = np.zeros((1,r))+0j
        Fzero = np.zeros((1,r))+0j
        Fninety = np.zeros((1,r))+0j
        
        #main loop II.9,II.10 completion of 2-D FFT
        for k in range(0,r):
            
            #p is the offset from center row or axis. this is necessary to computer
            #only the required points on the polar grid. The origin is included in 
            #every radial line and p changes sign after the origin point. 
            p = l
            if k == CTR:
                p = 0
            if k>CTR:
                p = -l    
            
            #n summation loop    
            for n in range(0,r):
                #II.9 Completion of 2-D FFT scaled X then Y axis
                Fx1[0,k]    = Fx1[0,k] +    fx[n,k]   *np.exp(-1j*2*np.pi*abs(C[k])*R[CTR-p]*beta*SN[n]/(l*(N+1)))
                Fx2[0,k] = Fx2[0,k] + fx[n,k]*np.exp(-1j*2*np.pi*abs(C[k])*R[CTR+p]*beta*SN[n]/(l*(N+1)))
                
                if (m%4==0) & (l == L):
                    pass
                else:
                    #II.10 Completion of 2-D FFT scaled Y then X axis
                    Fy1[0,k] = Fy1[0,k] + fy[k,n]*np.exp(-1j*2*np.pi*abs(R[k])*C[CTR+p]*beta*SN[n]/(l*(N+1)))
                    Fy2[0,k] = Fy2[0,k] + fy[k,n]*np.exp(-1j*2*np.pi*abs(R[k])*C[CTR-p]*beta*SN[n]/(l*(N+1)))
                    # 0 and 90 degree radial lines
                if l ==1:
                    Fzero[0,k] = Fzero[0,k] + fy[CTR,n]*np.exp(-1j*2*np.pi*C[k]*SN[n]/(l*(N+1)))
                    Fninety[0,k] = Fninety[0,k] + fx[n,CTR]*np.exp(-1j*2*np.pi*R[k]*SN[n]/(l*(N+1)))    
            
            # II.11 storing 4 radial lines, flips ensure storing in correct orientation (right to left)
            F[:,l] = np.fliplr(Fx1)
            F[:,m-l] = Fx2
            
            if (m%4==0) & (l == L):
                pass
            else:
                F[:,ACTR+l]= np.fliplr(Fy1)
                F[:,ACTR-l]= np.fliplr(Fy2)
            # Storing 0 and 90 degree radial lines
            if l ==1 :
               F[:,0] = np.fliplr(Fzero)
               F[:,ACTR] = np.fliplr(Fninety)
        print(l)
    return F
          
def adjoint(img):
    A = np.ones(11)
    return A

if __name__ == "__main__":
    Kino = fft(1)
    #flip the rows of the 2-D fft before shifting
    L = range(0,r)
    L = L[::-1]
    print(L)
    Kino = Kino[L,:]
    
    #fftshift to center around 0 frequency
    Kino = np.fft.fftshift(Kino, axes = (0,))
    print(Kino.shape)
    
    # perform ifft along each radial line to get radon transform (Fourier Slice Theorem)
    result = np.fft.ifft(Kino, axis = 0)
    Kino = pr.image(Kino, top_left = (0,1), bottom_right = (math.pi,-1))
    
    # Shift result:
    result = np.fft.ifftshift( result, axes = ( 0, ) )
    
    #adjust for padding
    result = result[PAD+1:-PAD+1,:]  
    print(result.shape)
    result = np.real( result )
    
    # Normalize:
    result /= ( 0.5 * ( r - 1 ) )
    
    #print
    result = pr.image(result, top_left = (0,1), bottom_right = (math.pi,-1))
    pp.imshow(result,cmap = 'gray_r', interpolation = 'nearest', 
		extent = ( result.top_left[ 0 ], result.bottom_right[ 0 ], result.bottom_right[ 1 ], result.top_left[ 1 ] ))
    pp.savefig('sample_sino.png')
    pp.show
