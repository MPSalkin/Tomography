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


def getBlockData2ComplementaryLines(linedata1,linedata2,beta_factor,sizeN_plus1):
    tiledLineData = np.ones((sizeN_plus1,1))*linedata1
    tiledLineDataConj = np.ones((sizeN_plus1,1))*linedata2
                               
    FrFTVarBlock1 = beta_factor * tiledLineData
    FrFTVarBlock2 = np.conjugate(beta_factor)*tiledLineDataConj
                                
    BlockDataFrom2Lines = FrFTVarBlock1 + FrFTVarBlock2
    return BlockDataFrom2Lines
    

def VectorizedFrFT_Centered(x,alpha):
    sizeX, DONTUSE = x.shape 
    N = sizeX - 1
    n1 = range(0,N+1)
    n2 = range(-N-1,0)
    n1 = np.array(n1)
    n2 = np.array(n2)
    n = np.concatenate((n1,n2),axis=0)
    n = n*np.array(np.ones((1,n.size)))
    n = np.transpose(n)
    M = np.array((range(0,N+1)))
    M = np.transpose(M*np.array(np.ones((1,M.size))))
    K = np.array((range(0,N+1)))
    K = np.transpose(K**np.array(np.ones((1,K.size))))
    Z = np.zeros((x.shape))
    M = M - N/2.0
    E_n = np.exp(-1j*np.pi*alpha*n**2/(N+1))
    
    PremultiplicationFactor = np.exp(1j*np.pi*M*N*alpha/(N+1))
    PostmultiplicationFactor = np.exp(1j*np.pi*alpha*N*K/(N+1))
    
    x_tilde = np.multiply(x, PremultiplicationFactor)
    x_tilde = np.concatenate((x_tilde,Z),axis=0)
    x_tilde = x_tilde*E_n
    
    FirstFFT = np.fft.fft(x_tilde, axis = 0) 
    SecondFFT = np.fft.fft(np.conj(E_n), axis = 0) 
    
    interimY= FirstFFT*SecondFFT
    interimY = np.transpose(interimY)
    interimY = np.fft.ifft(interimY)
    interimY= np.multiply(np.transpose(interimY), E_n)
    
    y = PostmultiplicationFactor * interimY[0:N+1,:]
    return y 


def  FRFT_Centered(x,alpha):
    x = x.flatten()
    N = x.size
    n = np.concatenate([np.linspace(0,N-1,N), np.linspace(-N,-1,N)])

    Factor1 = np.exp(-1j*np.pi*alpha*n**2)                                   # Equivalent to E(n)
    Factor2 = np.exp(1j*np.pi*np.linspace(-(N-1)/2, (N-1)/2, N)*(N-1)*alpha) # Equivalent to C(k')
    Factor3 = np.exp(1j*np.pi*np.linspace(0,N-1,N)*(N-1)*alpha)                    # Equivalent to B(k')

    x_tilde = x*Factor2                                                      #pre multiplication factor
    x_tilde = np.concatenate([x_tilde, np.zeros((N,))])
    x_tilde = x_tilde*Factor1

    
    XX = np.fft.fft(x_tilde, axis = 0)
    YY = np.fft.fft(np.conj(Factor1), axis = 0)
    
    y = np.fft.ifft(XX*YY, axis = 0)
    y = y*Factor1
    
    y = y[0:N]

    return y*Factor3

def pfft(A):
    
    #padding to nearest 2*(N+1)+1 <--- to make odd
    r1,c1 = A.shape
    global PAD, r
    PAD = int(np.ceil((r1*1.2 - r1)/2))
    #print(PAD)
    A = np.pad(A, PAD, 'minimum')
    r,c = A.shape
    
    #Set various endpoints and midpoints
    N = r-1 #N, as used in II.5,II.6,II.9,II.10 etc.
    M = r1-1 #number of radial lines (not including 180degrees)
    m = M # assigning m for use in loop
    CTR = (N+1)/2
    ACTR = M/2
    dtheta = np.pi/M 
    #print 'M=',M
    
    
    if np.mod(M,4) == 0:
        L = M/4
    elif np.mod(M,2) == 0:
        L = (M-2)/4
     
    #print(L)
    #print('^^Required levels^^')   
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
  
        for i in range(0,r):
            fx[i,:] = FRFT_Centered(A[i,:],alpha/(N+1))   
           
            if (np.mod(M,4)==0) and (l == L):
                pass 
            else:
                fy[:,i] = FRFT_Centered(A[:,-1-i],alpha/(N+1))
                
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
            
            if (np.mod(M,4)==0) & (l == L):
                pass
            else:
                F[:,ACTR+l]= (Fy1)
                F[:,ACTR-l]= (Fy2)
            # Storing 0 and 90 degree radial lines
            if l ==1 :
               F[:,0] = np.fliplr(Fzero)
               F[:,ACTR] = np.fliplr(Fninety)
        #print(l)
    return F
          
def apfft(Polar_Grid):
#    counts = pr.image_read( 'TomoData/noisyphantom/nslcounts.mat', dtype=np.float32 ) 
#    Polar_Grid = counts
#    Polar_Grid = np.ones((50,50))

    M, sizeN_plus1 = Polar_Grid.shape
    N = sizeN_plus1 - 1
    
    
    if np.mod(N,2) != 0:
        zeroPad = np.zeros((M,1))
        Polar_Grid = np.concatenate((Polar_Grid,zeroPad),1)
        M, sizeN_plus1 = Polar_Grid.shape
        N = sizeN_plus1 - 1
    

    L = (M-2)/4.0 # Number of levels 

    if np.mod(M-2,4)!=0 :
        L = np.ceil(L);                  
     
    lineSpacing = range(-N/2,N/2 + 1)
    lineSpacing = np.array(lineSpacing)
    lineSpacing = lineSpacing*np.array(np.ones((1,lineSpacing.size)))
    
    gridSpacing = np.multiply(np.transpose(lineSpacing),lineSpacing)
    
    ImageAdjoint = np.zeros((sizeN_plus1, sizeN_plus1)) # Suppose to be Complex Adjoint image
                           
    for l in range(1,int(L)+1): 
        #print l
        angle = l*180/M;
        alpha_l = math.cos(angle*math.pi/180)
        beta_l  = math.sin(angle*math.pi/180)

        beta_factor = np.exp((2*1j*np.pi*beta_l * gridSpacing)/ sizeN_plus1) # observe the scale change, it is +ve
        
        #Reversing the forward Polar Grid operation step by step
        #X-axis FrFT Grid
        #computing variable scale FrFT  first, gather the  lines oriented at angle from x-axis
        line1 = Polar_Grid[l, :]
        line2 = np.flip(Polar_Grid[M-l, :],0)
        
        FrFTVarBlock = getBlockData2ComplementaryLines(line1,line2,beta_factor,sizeN_plus1) 
        
        #computing uniform scale FrFT second
        FrFTVarBlock = np.transpose(FrFTVarBlock)        
        FrFTUniformBlock_x = VectorizedFrFT_Centered(FrFTVarBlock, -alpha_l)       
    
        ImageAdjoint = ImageAdjoint + FrFTUniformBlock_x
        ### COME BACK TO THIS . . . 
        
        if angle == 45: 
            continue # also works on python apparently 
        
        #Y-axis FrFT Grid
        line1 = Polar_Grid[M/2-l,:]
        line2 = Polar_Grid[M/2+l, :]
        FrFTVarBlock = getBlockData2ComplementaryLines( line1, line2,beta_factor,sizeN_plus1) 
        
        FrFTVarBlock = np.transpose(FrFTVarBlock)
        
        #computing uniform scale FrFT second
        FrFTUniformBlock_y = VectorizedFrFT_Centered(FrFTVarBlock, -alpha_l) 
        
        ImageAdjoint = ImageAdjoint + np.transpose(FrFTUniformBlock_y)
        
        #computing for zero and ninety degrees seperately
        if l == 1: 
            ZeroLine = Polar_Grid[0,:]
            NinetyLine = Polar_Grid[M/2,:]
            
            FrFTVarBlock1 =  np.ones((sizeN_plus1,1))*NinetyLine
            FrFTVarBlock2 =  np.ones((sizeN_plus1,1))*ZeroLine
            
            #computing uniform scale FrFT second
            FrFTUniformBlock_x = VectorizedFrFT_Centered(np.transpose(FrFTVarBlock2) , -1)          
            FrFTUniformBlock_y = VectorizedFrFT_Centered(np.transpose(FrFTVarBlock1) , -1)
            ImageAdjoint = ImageAdjoint + FrFTUniformBlock_x + np.transpose(FrFTUniformBlock_y)  
    return (np.transpose(np.flipud(ImageAdjoint)))






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
