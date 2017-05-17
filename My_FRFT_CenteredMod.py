
import numpy as np
def My_FRFT_CenteredMod(x,alpha):
    #=====================================================================
    # My Fractional Fourier Transform, computing the transform
    #               
    #            y[n]=sum_{k=0}^{N-1} x(k)*exp(-i*2*pi*k*n*alpha)  n=-N/2, ... ,N/2-1
    # 
    # For alpha=1/N we get the regular FFT, and for alpha=-1/N we get the regular IFFT. 
    #
    # Synopsis: y=My_FRFT_Centered(x,alpha)
    #
    # Inputs -    x - an N-entry vector to be transformed
    #                   alpha - the scaling factor (in the Pseudo-Polar it is in the range [-1/N,+1/N]
    # 
    # Outputs-  y - the transformed result as an N-entries vector
    #
    # Written by Michael Elad on March 20th 2005.
    # Adapted by  Jacob Cupul on April 1st 2017.
    #=====================================================================
    N = np.size(x)
    Factor2= np.exp(1j * np.pi *np.transpose(np.arange(N)) * N *1.0 * alpha)
    x_tilde=x*Factor2+0j

    n = np.transpose(np.concatenate((np.arange(N), np.arange(-N,0,1)),0))
    Factor = np.exp(-1j*np.pi*alpha*n**2.0)
    

    I = 1j*np.zeros((1,N))
    x_tilde = np.concatenate((x_tilde,I[0,:] ), 0)
    x_tilde = x_tilde*Factor    

    XX=np.fft.fft(x_tilde)
    YY=np.fft.fft(np.conj(Factor))
    y=np.fft.ifft(XX*YY)
    y=y*Factor
    y=y[0:N]*Factor2+0j
    return y