import numpy as np
import My_FRFT_CenteredMod as frft

def PPFFT(X):
#%=====================================================================
#% This function performs a Recto (Pseudo) Polar transform on a 2D signal X given on a 
#% Cartesian grid. If X is N*N, the output will have 2N by 2N output
#% values. 
#%   
#% Synopsis: Y=PPFFT(X)
#%
#% Inputs -    X      N*N matrix in cartesian grid, (n is assumed to be even)   
#% Outputs - Y      (2*N)*(2*N) matrix (theta,r)
#%
#% The following is an adaptation by the code written by Michael Elad.
#%
#%=====================================================================
    

    S1=1 
    S2=1
        
    #
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 1: Defining the sizes of the input/output arrays
    #%------------------------------------------------------------------------------------------------------------------------------
    
    S = np.shape(X)
    N1 = S[0]
    N = N1
    Y = np.zeros((2*S1*N,2*S2*N))*1j
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 2: Constructing quadrant 1 and 3
    #%------------------------------------------------------------------------------------------------------------------------------
    f_tilde=np.fft.fft(np.concatenate((X, 1j*np.zeros(((S1*2-1)*N,N))),0),axis=0);
    f_tilde=np.fft.fftshift(f_tilde,0); 
    f_tilde=np.concatenate((f_tilde, 1j*np.zeros((2*S1*N,N*(S2-1)))),1);
    
    for ll in range(-N*S1,N*S1,1):
         Y[ll+N*S1,range(N-1,-1,-1)]=(frft.My_FRFT_CenteredMod(f_tilde[ll+N*S1,:],ll/((N**2.0)*S1*S2)))
    #
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 3: Constructing quadrant 2 and 4
    #%------------------------------------------------------------------------------------------------------------------------------

    f_tilde=np.fft.fft(np.concatenate((X, 1j*np.zeros((N,(S1*2-1)*N))),1),axis=1); 
    f_tilde=np.fft.fftshift(f_tilde,1);
    f_tilde= np.transpose(f_tilde)
    f_tilde=np.concatenate((f_tilde, 1j*np.zeros((2*S1*N,N*(S2-1)))),1);
    
    for ll in range(-N*S1,N*S1,1):
         Y[ll+N*S1,range(S2*N,2*N*S2)]= (frft.My_FRFT_CenteredMod(f_tilde[ll+N*S1,:],ll/((N**2.0)*S1*S2)))
        
    return Y
    
    
def PPFFTBH(X):
#%=====================================================================
# This function is the BH component of  PPFFT
#%=====================================================================
    

    S1=1 
    S2=1
        
    #
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 1: Setting up input and output arrays
    #%------------------------------------------------------------------------------------------------------------------------------
    
    S = np.shape(X)
    N1 = S[0]
    N = N1
    Y = np.zeros((2*S1*N,S2*N))*1j
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 2: Constructing quadrant 1 and 3
    #%------------------------------------------------------------------------------------------------------------------------------
    f_tilde=np.fft.fft(np.concatenate((X, 1j*np.zeros(((S1*2-1)*N,N))),0),axis=0);
    f_tilde=np.fft.fftshift(f_tilde,0); 
    f_tilde=np.concatenate((f_tilde, 1j*np.zeros((2*S1*N,N*(S2-1)))),1);
    
    for ll in range(-N*S1,N*S1,1):
         Y[ll+N*S1,range(N-1,-1,-1)]=(frft.My_FRFT_CenteredMod(f_tilde[ll+N*S1,:],ll/((N**2.0)*S1*S2)))

        
    return Y  
    
def PPFFTBV(X):
#
#%=====================================================================
#This function is the BV component of  PPFFT
#%=====================================================================
    S1=1 
    S2=1  
    #
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 1: Setting up input and output arrays
    #%------------------------------------------------------------------------------------------------------------------------------
    
    S = np.shape(X)
    N1 = S[0]
    N = N1
    Y = np.zeros((2*S1*N,S2*N))*1j
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 3: Constructing quadrant 2 and 4
    #%------------------------------------------------------------------------------------------------------------------------------

    f_tilde=np.fft.fft(np.concatenate((X, 1j*np.zeros((N,(S1*2-1)*N))),1),axis=1); 
    f_tilde=np.fft.fftshift(f_tilde,1);
    f_tilde= np.transpose(f_tilde)
    f_tilde=np.concatenate((f_tilde, 1j*np.zeros((2*S1*N,N*(S2-1)))),1);
    
    for ll in range(-N*S1,N*S1,1):
         Y[ll+N*S1,range(0,N)]= (frft.My_FRFT_CenteredMod(f_tilde[ll+N*S1,:],ll/((N**2.0)*S1*S2)))
        
    return Y
    
    
def APPFFT(X):
#
#    %====================================================================
#    % This function performs a Recto (Pseudo) Polar ADJOINT transform on a 2D signal X 
#    % given on the PP-coordinate system. If X is 2N*2N, the output will have N by N values. 
#    % The algorithm applied here uses the Fast Fractional Fourier Transform. 
#    %   
#    % Synopsis: Y=APPFFT(X)
#    %
#    % Inputs -    X     2N*2N matrix in pseudo-polar grid, (N is assumed to be even)           
#    % Outputs - Y      N*N matrix (Cartesian grid)    
#    % 
#    %The following is an adaptation by the code written by Michael Elad.
#    %
#    %====================================================================
    
    # preliminary checks of the input size
    S = np.shape(X); 
    N = S[0]
    m2 = S[1]
    if N!=m2:
        print('Non Square input array');
        return
    elif np.floor(N/4)-N/4!=0:
        print('Size of the input array is not an integer mul. by 4');
        return
        
        
    N=N/2;
      
    Y=np.zeros((N,N))*0j
    #FrFT for the BV rays
    Temp1=np.zeros((N,2*N))*0j
    for l in range(-N,N):
        Xvec= X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneLine=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp1[:,l+N]=np.transpose(OneLine);
    #ifft along complementary axis for BV rays and store output
    Temp_Array=2*N*np.fft.ifft(Temp1,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Y=np.transpose(Temp_Array)
    #FrFT for the BH rays
    Temp2=np.zeros((N,2*N))*0j;
    for l in range(-N,N):
        Xvec=X[2*N-1:N-1:-1,l+N]
        alpha=-l/N**2.0
        OneCol=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp2[:,l+N]=OneCol
    #ifft along complementary axis for BH rays and store output
    Temp_Array= 2*N*np.fft.ifft(Temp2,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Z=Temp_Array
    #Re-orient and sum output.
    return (np.flipud(Z)+Y)
    
def APPFFTBH(X):
#
#    %====================================================================
#       This is the adjoint operator for only the BH rays
#    %====================================================================
    
    # preliminary checks of the input size
    S = np.shape(X); 
    N = S[0]
    m2 = S[1]
    if N!=m2:
        pass
    elif np.floor(N/4)-N/4!=0:
        print('Size of the input array is not an integer mul. by 4');
            
    Y=np.zeros((N,N))*0j
    
    Temp1=np.zeros((N,2*N))*0j
    #Start with FrFT along axis
    for l in range(-N,N):
        Xvec= X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneLine=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp1[:,l+N]=np.transpose(OneLine);
    #Finish with ifft along complementary axis
    Temp_Array=2*N*np.fft.ifft(Temp1,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Y=np.transpose(Temp_Array)
    
    return Y    
    
def APPFFTBV(X):
#
#    %====================================================================
#       This is the adjoint operator for only the BV rays
#    %====================================================================

    # preliminary checks of the input size
    S = np.shape(X); 
    N = S[0]
    m2 = S[1]
    if N!=m2:
        pass

    elif np.floor(N/4)-N/4!=0:
        print('Size of the input array is not an integer mul. by 4');

    Z=np.zeros((N,N))*0j
    
    Temp2=np.zeros((N,2*N))*0j;
    #Start with FrFT along axis
    for l in range(-N,N):
        Xvec=X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneCol=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp2[:,l+N]=OneCol
    #Finish with ifft along complementary axis
    Temp_Array= 2*N*np.fft.ifft(Temp2,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Z=Temp_Array
    
    return np.flipud(Z)