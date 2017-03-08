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
    nfft_zp_factor = 1.2
    A = np.ones((15,15))
    #A = pr.image_read( 'TomoData/PhantomData/sl.mat')
    r,c = A.shape
    global PAD 
    PAD = (r+3)/2
    print(PAD)
    A = np.pad(A, PAD, 'minimum')
    r,c = A.shape
    
    N = r-1
    M = r-1
    m = M
    dtheta = np.pi/M
    F = np.zeros((r,M))+0j
    
    if np.mod(M,4) == 0:
        L = M/4
    elif np.mod(M,2) == 0:
        L = (M-2)/4

    fx = np.zeros((r,c))+0j
    fy = np.zeros((r,c))+0j   
    CTR = (N+1)/2
    ACTR = M/2
    print(ACTR)
    C = range(-N/2,N/2 + 1)
    R = C[::-1]
    SN = C 

    for l in range(1,L+1):
        # scaling by alpha_l
        alpha = np.cos(l*dtheta)
        for i in range(0,r) :
            for j in range (0,c) :
                fx[i,j]= 0
                for n in range(0,r):
                    #II.5 FrFT along rows
                    fx[i,j] = fx[i,j] + A[i,n]*np.exp(-1j*2*np.pi*C[j]*alpha*SN[n]/(N+1))
        
        for j in range (0,c) :  
            for i in range(0,r) :
                fy[i,j] = 0
                for n in range(0,c):
                    #II.6 FrFT along columns
                    fy[i,j] = fy[i,j] + A[n,j]*np.exp(-1j*2*np.pi*R[i]*alpha*SN[n]/(N+1))
       
        #Scaling by Beta_l
        beta = np.sin(l*dtheta)
        Fx1 = np.zeros((1,r))+0j
        Fx2 = np.zeros((1,r))+0j
        Fy1 = np.zeros((1,r))+0j
        Fy2 = np.zeros((1,r))+0j
        Fzero = np.zeros((1,r))+0j
        Fninety = np.zeros((1,r))+0j
        for k in range(0,r):
            p = l
            if k == CTR:
                p = 0
            if k>CTR:
                p = -l    
            for n in range(0,r):
                #II.9 Completion of 2-D FFT scaled X then Y axis
                Fx1[0,k]    = Fx1[0,k] +    fx[n,k]   *np.exp(-1j*2*np.pi*abs(C[k])*R[CTR+p]*beta*SN[n]/(l*(N+1)))
                Fx2[0,-k-1] = Fx2[0,-k-1] + fx[n,-k-1]*np.exp(-1j*2*np.pi*abs(C[-k-1])*R[CTR-p]*beta*SN[n]/(l*(N+1)))
                
                #II.10 Completion of 2-D FFT scaled Y then X axis
                Fy1[0,k] = Fy1[0,k] + fy[k,n]*np.exp(-1j*2*np.pi*abs(R[k])*C[CTR+p]*beta*SN[n]/(l*(N+1)))
                Fy2[0,k] = Fy2[0,k] + fy[k,n]*np.exp(-1j*2*np.pi*abs(R[k])*C[CTR-p]*beta*SN[n]/(l*(N+1)))
                # 0 and 90 degree radial lines
                if l ==1:
                    Fzero[0,k] = Fzero[0,k] + fy[CTR,n]*np.exp(-1j*2*np.pi*C[k]*SN[n]/(l*(N+1)))
                    Fninety[0,k] = Fninety[0,k] + fx[n,CTR]*np.exp(-1j*2*np.pi*R[k]*SN[n]/(l*(N+1)))    
            
            # II.11 storing 4 radial lines
            F[:,l] = Fx1
            F[:,m-l] = Fx2
            F[:,ACTR+l]= Fy1
            F[:,ACTR-l]= Fy2
            # Storing 0 and 90 degree radial lines
            if l ==1 :
               F[:,0] = Fzero
               F[:,ACTR] = Fninety
            print(l)
    return F
          
def adjoint(img):
    A = np.ones(11)
    return A

if __name__ == "__main__":
    Kino = fft(1)
    Kino = np.fft.fftshift(Kino, axes = (0,))
    print(Kino.shape)
    Kino = pr.image(Kino, top_left = (0,1), bottom_right = (math.pi,-1))
    result = np.fft.ifft(Kino, axis = 0)
    Kino = pr.image(Kino, top_left = (0,1), bottom_right = (math.pi,-1))
    # Shift result:
    result = np.fft.ifftshift( result, axes = ( 0, ) )
    result = result[PAD+1:-PAD+1,:]  
    print(result.shape)
    result = np.real( result )
    # Normalize:
    result /= ( 0.5 * ( 11 - 1 ) )
    #print
    result = pr.image(result, top_left = (0,1), bottom_right = (math.pi,-1))
    pp.imshow(result,cmap = 'gray_r', interpolation = 'nearest', 
		extent = ( result.top_left[ 0 ], result.bottom_right[ 0 ], result.bottom_right[ 1 ], result.top_left[ 1 ] ))
    pp.savefig('sample_sino.png')
    pp.show
