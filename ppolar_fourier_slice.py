#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:46:22 2017

@author: CupulFamily
"""


import pyraft as pr
import numpy as np
import math
import warnings
import matplotlib.pyplot as pp
import pfft as polar
#import P_INTERP_PP as PIPP

# TODO: Take sinogram's t_extent and image's corners in
# consideration;

def make_fourier_slice_radon_transp( sino, shape = None):
   """
   make_fourier_slice_radon_transp( shape, sino, sino_zp_factor = 1.5, nfft_zp_factor = 1.2, nfft_m = 2 )

   Creates routines for computing the discrete Radon Transform and its conjugate.

   Parameters:

   sino           : Sinogram. Data is unimportant, but size and ranges are not.
   shape          : Shape of the backprojection result (same of projection argument)
                    You can provide an image instead, in which case its FOV must
                    be the unit square.


   Returns:

   return fast_radon, fast_radon_transpose

   functions for computing the Radon transform and its numerical transpose.

   Usage:

  """

   if shape is None:
      shape = ( np.min(sino.shape)+1, np.min(sino.shape)+1)
   img = pr.image( shape ) #consider dividing by 2

   if ( sino.top_left[ 1 ] != 1.0 ) or \
      ( sino.bottom_right[ 1 ] != -1.0 ):
         raise ValueError( 'Invalid sinogram t range. Must be ( -1.0, 1.0 )' )
   if ( img.top_left != ( -1.0, 1.0 ) ) or \
      ( img.bottom_right != ( 1.0, -1.0 ) ):
         print img.top_left, img.bottom_right
         raise ValueError( 'Invalid image range. Must be ( -1.0, 1.0 ) x ( -1.0, 1.0 ).' )
   if ( img.shape[ 0 ] != sino.shape[ 0 ] ) or \
      ( img.shape[ 1 ] != sino.shape[ 0 ] ):
         warnings.warn( 'Attention: img and sino should preferably have matching dimensions.' )
   r1 = img.shape[0] 
   PAD = int(np.ceil((r1*1.2 - r1)/2))
   # Padded sinogram size (make it even):
   #sino_padded_size = int( math.ceil( sino_zp_factor * sino.shape[ 0 ] ) )
   #sino_padding_size = sino_padded_size - sino.shape[ 0 ]
 
   
   Forward = polar.pfft
   Adjoint = polar.apfft

   def fast_radon( img ):
      """
         Compute projection through projection-slice theorem
      """

      # Execute polar polar 2D-FFT:
      fsino = Forward(img)
      
      
      fsino = np.fft.ifftshift(fsino)
      # Inverse FFT:
      Kino = pr.image( fsino, top_left = sino.top_left, bottom_right = sino.bottom_right )
      pp.imshow(Kino,cmap = 'Greys', interpolation = 'nearest', 
               extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
      
          
      result = np.fft.ifft( fsino, axis = 0 )

      # Shift result:
      result = np.fft.fftshift(result)

      # Get real part:
      result = np.real(result)

      # Normalize:
      result = result[PAD-1:-PAD-1,:] 
      result /= ( 0.5 * ( sino.shape[ 0 ] - 1 ) )
      
          
      return pr.image( result, top_left = sino.top_left, bottom_right = sino.bottom_right )

   def fast_radon_transpose( sino ):
      """
         Compute backprojection through projection-slice theorem
      """

      # Zero-pad sinogram:
      #fsino = np.zeros( ( sino_padded_size, sino.shape[ 1 ] ) )
      #fsino[ delta + odd_sino : fsino.shape[ 0 ] - delta - extra + odd_sino ] = sino
      fsino = sino
      
      # Shift sinogram columns:
      fsino = np.fft.fftshift(fsino)

      # Fourier transform of projections:
      fsino = np.fft.fft( fsino, axis = 0 )
      
     
      fsino = np.fft.fftshift(fsino)
      
      # Compute adjoint of 2D polar Polar Fourier transform:
      Kino = pr.image( fsino, top_left = img.top_left, bottom_right = img.bottom_right )
      pp.imshow(Kino,cmap = 'Greys', interpolation = 'nearest', 
               extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
      
      pp.show    
    
      result = Adjoint(np.transpose(fsino))
      
      # Get real part:
      result = np.real( result )


      result /= ( 0.5 * (sino.shape[ 0 ]-1) * ( img.shape[ 1 ] - 1 ) )
      

      return pr.image( result, top_left = img.top_left, bottom_right = img.bottom_right )

   return fast_radon, fast_radon_transpose

if __name__ == '__main__':
    
   import h5py
   f = h5py.File( '/home/elias/curimatã/CM-Day3/tomoTesteH2O.h5', 'r' )
   v = f[ 'images' ]
   sino = pr.image( np.transpose( v[ :, 450, : ] ).astype( np.float64 ), x_extent = ( 0.0, math.pi ) )
   fast_radon, fast_transp = make_fourier_slice_radon_transp( sino )

