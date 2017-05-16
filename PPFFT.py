import numpy as np
import My_FRFT_CenteredMod as frft

def PPFFT(X):
#
#%=====================================================================
#% This function performs a Recto (Pseudo) Polar transform on a 2D signal X given on a 
#% Cartesian grid. If X is N*N, the output will have 2NS1 by 2NS2 output
#% values. If the input is not square, it is squared before the transform.
#% Also, the size is increased so as to get even number of rows and columns.
#%   
#% Synopsis: Y=PPFFT(X,S1,S2)
#%
#% Inputs -    X      N*N matrix in cartesian grid, (n is assumed to be even)   
#%                  S1   oversampling factor along the rays
#%                  S2   oversampling factor along the slopes (e.g. angles)
#% Outputs - Y      (2*S1*N)*(2*S2*N) matrix (theta,r)
#%
#% Example: 
#%       This shows that the obtained transform matches the brute-force one
#%       N=16; X=randn(N,N); X(N/4:3*N/4,N/4:3*N/4)=1;
#%       [XC,YC]=Create_Oversampled_Grid('D',[N,pi,5,3],'.r');
#%       tic; Y_slow=Brute_Force_Transform(X,XC,YC); toc;
#%       tic; Y_fast=PPFFT(X,5,3); toc;
#%       figure(1); clf;
#%       subplot(2,2,1); imagesc(real(Y_fast)); xlabel('Fast - real part');
#%       subplot(2,2,2); imagesc(imag(Y_fast)); xlabel('Fast - imaginary part');
#%       subplot(2,2,3); imagesc(real(Y_slow)); xlabel('Slow - real part');
#%       subplot(2,2,4); imagesc(imag(Y_slow)); xlabel('Slow - imaginary part');
#%       disp(['Error between fast and slow: ',num2str(max(abs(Y_fast(:)-Y_slow(:))))]);
#%
#%       This shows the relation between the PPFFT and the matrix transform 
#%       N=16; X=randn(N,N)+sqrt(-1)*randn(N,N);
#%       Yf=PPFFT(X,1,1); 
#%       [xc,yc]=Create_Oversampled_Grid('D',[N,pi,1,1],'');
#%       T=Transform_Matrix(N,N,xc,yc);
#%       Yc=T*X(:);
#%       plot([real(Yf)-real(reshape(Yc,[32,32]))']); % there is a transpose relation
#%       plot([imag(Yf)-imag(reshape(Yc,[32,32]))']); % there is a transpose relation
#% 
#% Written on March 20th, 2005 by Michael Elad.
#%=====================================================================
    

    S1=1 
    S2=1
        
    #
    #%------------------------------------------------------------------------------------------------------------------------------
    #% Stage 1: Checking input size and defining the sizes of the input/output arrays
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
#
#%=====================================================================
#
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
#
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
#    % Example: 
#    % 
#    %   The following is a way to verify that this works as an adjoint -
#    %   choosing random X and Y, one must obtain that <y,Ax> = <x,adj(A)y>'
#    %  
#    %   N=16; X=randn(N)+sqrt(-1)*randn(N); Y=randn(2*N)+sqrt(-1)*randn(2*N);
#    %   AX=PPFFT(X);
#    %   AtY=APPFFT(Y);
#    %   disp(abs( sum(sum(Y'.*AX)) -conj(sum(sum(X'.*AtY)))));       
#    % 
#    % Written on March 20th, 2005 by Michael Elad.
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
    
    Temp1=np.zeros((N,2*N))*0j
    for l in range(-N,N):
        Xvec= X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneLine=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp1[:,l+N]=np.transpose(OneLine);
    
    Temp_Array=2*N*np.fft.ifft(Temp1,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Y=np.transpose(Temp_Array)
    
    Temp2=np.zeros((N,2*N))*0j;
    for l in range(-N,N):
        Xvec=X[2*N-1:N-1:-1,l+N]
        alpha=-l/N**2.0
        OneCol=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp2[:,l+N]=OneCol

    Temp_Array= 2*N*np.fft.ifft(Temp2,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Z=Temp_Array
    
    return (np.flipud(Z)+Y)
    
def APPFFTBH(X):
#
#    %====================================================================
#
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
    for l in range(-N,N):
        Xvec= X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneLine=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp1[:,l+N]=np.transpose(OneLine);
    
    Temp_Array=2*N*np.fft.ifft(Temp1,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Y=np.transpose(Temp_Array)
    
    return Y    
    
def APPFFTBV(X):
#
#    %====================================================================
#
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
    for l in range(-N,N):
        Xvec=X[N-1::-1,l+N]
        alpha=-l/N**2.0
        OneCol=frft.My_FRFT_CenteredMod(Xvec,alpha)
        Temp2[:,l+N]=OneCol

    Temp_Array= 2*N*np.fft.ifft(Temp2,axis=1)
    Temp_Array= np.dot(Temp_Array[:,0:N],np.diag((-1.0)**np.arange(0,N)))
    Z=Temp_Array
    
    return np.flipud(Z)