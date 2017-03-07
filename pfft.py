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
    A = np.ones((11,11))
    #A = pr.image_read( 'TomoData/PhantomData/sl.mat')
#    PAD = ([i * nfft_zp_factor for i in A.shape])
#    
#    PAD[0] = int(PAD[0])
#    PAD[1] = int(PAD[1])
#    print(PAD)
#    A = np.pad(A, PAD, 'minimum')
    r,c = A.shape
    N = r-1
    M = 10
    m = M
    dtheta = np.pi/M
    F = np.zeros((M,r))+0j
    
    if np.mod(M,4) == 0:
        L = M/4
    elif np.mod(M,2) == 0:
        L = (M-2)/4

    fx = np.zeros((r,c))+0j
    fy = np.zeros((r,c))+0j   
    C = (N+1)/2
    C2 = M/2
    for l in range(1,L+1):
        # scaling by alpha_l
        alpha = np.cos(l*dtheta)
        for j in range (0,c) :
            for i in range(0,r) :
                fx[i,j]= 0
                for n in range(0,r):
                    #II.5 FrFT along rows
                    fx[i,j] = fx[i,j] + A[i,n]*np.exp(-1*1j*2*np.pi*(i-N/2)*alpha*(n-N/2)/(N+1))
       
        for j in range (0,c) :
            for i in range(0,r) :
                fy[i,j] = 0
                for n in range(0,c):
                    #II.6 FrFT along columns
                    fy[j,i] = fy[j,i] + fx[n,i]*np.exp(-1*1j*2*np.pi*(i-N/2)*alpha*(n-N/2)/(N+1))
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
            if k == C:
                p = 0
            
            ps = 0            
            for n in range(0,r):
                #II.9 Completion of 2-D FFT scaled X then Y axis
                Fx1[0,-k-1] = Fx1[0,-k-1] + fx[n,-k-1]*np.exp(-1*1j*2*np.pi*abs((c-k-1)-N/2)*((C+p)-N/2)*beta*(n-N/2)/(l*(N+1)))
                Fx2[0,k] = Fx2[0,k] + fx[n,k]*np.exp(-1*1j*2*np.pi*abs(k-N/2)*((C-p)-N/2)*beta*(n-N/2)/(l*(N+1)))
                
                #II.10 Completion of 2-D FFT scaled Y then X axis
                Fy1[0,k] = Fy1[0,k] + fy[k,n]*np.exp(-1*1j*2*np.pi*abs(k-N/2)*((C+p)-N/2)*beta*(n-N/2)/(l*(N+1)))
                Fy2[0,k] = Fy2[0,k] + fy[k,n]*np.exp(-1*1j*2*np.pi*abs(k-N/2)*((C-p)-N/2)*beta*(n-N/2)/(l*(N+1)))
                # 0 and 90 degree radial lines
                if l ==1:
                    Fzero[0,-k-1] = Fzero[0,-k-1] + fx[n,-k-1]*np.exp(-1*1j*2*np.pi*abs((c-k-1)-N/2)*((C+ps)-N/2)*beta*(n-N/2)/(l*(N+1)))
                    Fninety[0,k] = Fninety[0,k] + fy[k,n]*np.exp(-1*1j*2*np.pi*abs(k-N/2)*((C+ps)-N/2)*beta*(n-N/2)/(l*(N+1)))    
            
            # II.11 storing 4 radial lines
            F[l,:] = Fx1
            F[m-l,:] = Fx2
            F[C2+l,:]= Fy1
            F[C2-l,:]= Fy2
            # Storing 0 and 90 degree radial lines
            if l ==1 :
               F[0,:] = Fzero
               F[C2,:] = Fninety
        print(l)
    return pr.image(F, top_left = (0,1), bottom_right = (math.pi,-1))
          
def adjoint(img):
    A = np.ones(11)
    return A

if __name__ == "__main__":
    Kino = fft(1)
    Kino = np.transpose(Kino)
    result = np.fft.ifft(Kino, axis = 0)
      #print(result.shape)
    # Shift result:
    result = np.fft.ifftshift( result, axes = ( 0, ) )
    result = np.real( result )
    # Normalize:
    result /= ( 0.5 * ( 10 - 1 ) )
    #print
    pp.imshow(result,cmap = 'gray_r', interpolation = 'nearest', 
		extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
    pp.savefig('sample_sino.png')
    pp.show
