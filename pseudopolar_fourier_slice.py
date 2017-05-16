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
import PPFFT as pseudo
import P_INTERP_PP as PIPP

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
      shape = ( np.max(sino.shape)/2, np.max(sino.shape[ 0 ])/2 )
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


   # Padded sinogram size (make it even):
   #sino_padded_size = int( math.ceil( sino_zp_factor * sino.shape[ 0 ] ) )
   #sino_padding_size = sino_padded_size - sino.shape[ 0 ]
   if (sino.top_left[0] == 0 and sino.bottom_right[0] > np.pi/2):
       Forward = pseudo.PPFFT
       Adjoint = pseudo.APPFFT
       Toggle = 0
   elif sino.bottom_right[0] < np.pi:
       Forward = pseudo.PPFFTBV
       Adjoint = pseudo.APPFFTBV
       Toggle = 1
   else:
       Forward = pseudo.PPFFTBH
       Adjoint = pseudo.APPFFTBH
       Toggle = 1
    


   def fast_radon( img ):
      """
         Compute projection through projection-slice theorem
      """

      # Execute pseudo polar 2D-FFT:
      fsino = Forward(img)
      
      
      fsino = np.fft.fftshift(fsino)
      # Inverse FFT:
      result = np.fft.ifft( fsino, axis = 0 )

      # Shift result:
      result = np.fft.ifftshift(result, axes = ( 0, ))

      # Get real part:
      result = np.real(result)

      # Normalize:
          
      result /= ( 0.5 * ( sino.shape[ 0 ] - 1 ) )
      
#      Kino = pr.image(result, top_left = sino.top_left, bottom_right = sino.bottom_right )
#      
#      pp.imshow(Kino,cmap = 'plasma', interpolation = 'nearest', 
#		extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
#      pp.show
      
      # Return image with appropriate bounding box:
      if Toggle == 1:
          result = np.fft.fftshift(result,1)
          
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
      fsino = np.fft.fftshift( fsino, axes = ( 0, ) )

      # Fourier transform of projections:
      fsino = np.fft.fft( fsino, axis = 0 )

      
      fsino = np.fft.fftshift(fsino)
      
      pp.show
      # Compute adjoint of 2D Pseudo Polar Fourier transform:
      if Toggle == 1:
          fsino = np.fft.fftshift(fsino,1)
#      Kino = pr.image( fsino, top_left = img.top_left, bottom_right = img.bottom_right )
#      pp.imshow(Kino,cmap = 'plasma', interpolation = 'nearest', 
#		extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
      result = Adjoint(np.transpose(fsino))
      
      # Get real part:
      result = np.real( result )

#      Kino = pr.image( result, top_left = img.top_left, bottom_right = img.bottom_right )
#      pp.imshow(Kino,cmap = 'plasma', interpolation = 'nearest', 
#		extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
#      pp.show
      # Normalize:( sino_padded_size - 1 )
      result /= ( 0.5 * (sino.shape[ 0 ]-1) * ( img.shape[ 1 ] - 1 ) )
      

      return pr.image( result, top_left = img.top_left, bottom_right = img.bottom_right )

   return fast_radon, fast_radon_transpose

if __name__ == '__main__':
    
   import h5py
   f = h5py.File( '/home/elias/curimatã/CM-Day3/tomoTesteH2O.h5', 'r' )
   v = f[ 'images' ]
   sino = pr.image( np.transpose( v[ :, 450, : ] ).astype( np.float64 ), x_extent = ( 0.0, math.pi ) )
   fast_radon, fast_transp = make_fourier_slice_radon_transp( sino )

