import numpy as np
import pyraft as pr
import matplotlib.pyplot as pp
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

def P_INTERP_PP(In1):
#%=====================================================================
#% This function performs interpolation from the Polar-Grid to the
#% Pseudo-Polar domain for use in the adjoint. The interpolation is in
#% two stages first along each ray to obtain non-equispaced points, then
#% along each row to obtain non-equiangular rays on the concentric square
#% grid.
#%   In - 2D-FFT on the Polar Grid (r, theta)
#%   Out- 2D-FFT on the Pseudo-Polar Grid (r, theta)
#%=====================================================================

    if In1 is None:
        In1 = pr.image_read('TomoData/noisyphantom/nslcounts.mat',dtype=np.float32)
   
    OS1=1;
    OS2=1;
    K = np.shape(In1)
    In =  In1
    print('import done')
    
    N = min(np.shape(In)); #assuming square input

    # C. resampling the rays to the Pseudo-Polar locations
    F=In[:,0:N/2]
    Fout = np.zeros((OS2*N,N/2))*0j
    Xcor = np.arange(-2.0*np.pi,2.0*np.pi-np.pi/(N*OS2),2.0*np.pi/(N*OS2))   #these are the current locations
    Xcor1 = np.arange(-np.pi*1.0,np.pi-np.pi/(2*N*OS2),2.0*np.pi/(2.0*N*OS2))  
    print('initialization done')
    for k in range(1,N/2+1):
        Ray=np.transpose(F[:,k-1])
        Factor=np.cos((k-N/4.0)*np.pi/(N)); # completion of current locations
        #f2 = interp1d(Xcor, Ray, kind='linear')
        f2 = CubicSpline(Xcor,Ray)
        Fout[:,k-1]=np.transpose(f2(Xcor1[0::2*OS2]/Factor));
  
    # C. resampling the rays to the Pseudo-Polar locations
    G=In[:,N/2::];
    Gout = np.zeros((N,N/2))*0j;
    Ycor = np.arange(-2.0*np.pi,2.0*np.pi-np.pi/(N*OS2),2.0*np.pi/(N*OS2)) #these are the current locations
    Ycor1 = np.arange(-np.pi*1.0,np.pi-np.pi/(2*N*OS2),2.0*np.pi/(2.0*N*OS2))  
    for k in range(1,N/2+1):
        Ray=G[:,k-1]
        Factor=np.cos((k-N/4.0 )*np.pi/(N)); # completion of current locations
        #f2 = interp1d(Xcor, Ray,kind='linear')
        f2 = CubicSpline(Ycor,Ray)
        Gout[:,k-1]=np.transpose(f2(Ycor1[0::2*OS2]/Factor)) 

    # B. row/columnwise interpolation to obatain equi-distant slope
    Xcor1 = 2.0*np.pi/(N*OS2)*np.arange(-OS1*N/2.0,OS1*N/2.0-1,2)/OS1/N
    Xcor2 = 1.0*np.pi/(N*OS2)*np.tan(np.pi*np.arange(-N/2.0,N/2.0-1,2)/N/2.0)
    steps = range(-N/2*OS2,N/2*OS2)
    for ll in steps:
        Temp1=Fout[ll+OS2*N/2,:]
        if ll != 0:
            #f2 = interp1d(Xcor2*ll,Temp1,kind='linear', fill_value='extrapolate')
            if ll > 0:
                f2 = CubicSpline(Xcor2*ll,Temp1)
                Fout[ll+OS2*N/2,0:N/2]=f2(Xcor1*ll);
            else:
                f2 = CubicSpline(np.flip(Xcor2*ll,0),np.flip(Temp1,0))
                Fout[ll+OS2*N/2,0:N/2]=f2(Xcor1*ll);
        else:
            Fout[ll+OS2*N/2,0:N/2]=Temp1[0::OS1];
        
    # B. row/columnwise interpolation to obatain equi-distant slope  
    Ycor1 = 2.0*np.pi/(N*OS2)*np.arange(-OS1*N/2.0,OS1*N/2.0-1,2)/OS1/N
    Ycor2 = 1.0*np.pi/(N*OS2)*np.tan(np.pi*np.arange(-N/2.0,N/2.0-1,2)/N/2.0)
    steps = range(-N/2*OS2,N/2*OS2)
    for ll in steps:
        Temp2=Gout[ll+OS2*N/2,:];
        if ll !=0:
            #f2 = interp1d(Ycor2*ll,Temp1, kind='linear',fill_value='extrapolate')
            if ll > 0:
                f2 = CubicSpline(Ycor2*ll,Temp2)
                Gout[ll+OS2*N/2,0:N/2]= f2(Ycor1*ll)
            else:
                f2 = CubicSpline(np.flip(Ycor2*ll,0),np.flip(Temp2,0))
                Gout[ll+OS2*N/2,0:N/2]=f2(Ycor1*ll);
                    
        else:
            Gout[ll+OS2*N/2,0:N/2]= Temp2[0::OS1];

    Out = np.concatenate(((Fout), Gout),1)
    return Out

    
#padding function for purposes of interpolation (requires 2MxM intput into interp)    
def Pad2x(In1):
    #value = np.min(In1)
    value = np.average(In1[0,:])
    #value=0
    K = np.shape(In1)
    In2 = np.concatenate((np.flipud(In1[:,3*K[1]/4::]),In1[:,0:3*K[1]/4] ),1)
    if K[0] < 2*K[1]:
        temp = np.random.poisson(np.ones((2*K[1],K[1]))*value)+0j
        temp[-K[0]/2+K[1]:K[0]/2+K[1],:] = In2
        Out = temp
    else:
        Out = In1
    #print('padding done')  
    return Out