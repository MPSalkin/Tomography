# I use the one encoding: utf8
import pyraft as pr
import numpy as np
import math
import pynfft.nfft as nf
import warnings
import matplotlib.pyplot as pp

# TODO: Take sinogram's t_extent and image's corners in
# consideration;

def make_fourier_slice_radon_transp( sino, shape = None, sino_zp_factor = 1.5, nfft_zp_factor = 1.2, nfft_m = 2 ):
   """
   make_fourier_slice_radon_transp( shape, sino, sino_zp_factor = 1.5, nfft_zp_factor = 1.2, nfft_m = 2 )

   Creates routines for computing the discrete Radon Transform and its conjugate.

   Parameters:

   sino           : Sinogram. Data is unimportant, but size and ranges are not.
   shape          : Shape of the backprojection result (same of projection argument)
                    You can provide an image instead, in which case its FOV must
                    be the unit square.
   sino_zp_factor : Zero-padding factor for Fourier transform of projection
   nfft_zp_factor : Zero-padding factor for Fourier transform of image
   nfft_m         : Number of summation terms in nfft series.

   Returns:

   return fast_radon, fast_radon_transpose

   functions for computing the Radon transform and its numerical transpose.

   Usage:

import pyraft as pr
import matplotlib.pyplot as pp
import h5py
import time

f = h5py.File( '/home/elias/curimatã/CM-Day3/tomoTesteH2O.h5', 'r' )
v = f[ 'images' ]
sino = pr.image( np.transpose( v[ :, 450, : ] ).astype( np.float64 ), x_extent = ( 0.0, math.pi ) )

fast_radon, fast_transp = make_fourier_slice_radon_transp( sino )

st = time.time()
bp = fast_transp( sino )
print 'Done!', time.time() - st

st = time.time()
bp2 = pr.radon_transpose( sino, np.zeros( bp.shape ) )
print 'Done!', time.time() - st

pp.imshow( bp, interpolation = 'nearest' ); pp.colorbar(); pp.show()
pp.imshow( bp2, interpolation = 'nearest' ); pp.colorbar(); pp.show()

st = time.time()
rbp = fast_radon( bp )
print 'Done!', time.time() - st

st = time.time()
rbp2 = pr.radon( bp, pr.image( np.zeros( sino.shape ), x_extent = ( 0.0, math.pi ) ) )
print 'Done!', time.time() - st

pp.imshow( rbp, interpolation = 'nearest' ); pp.colorbar(); pp.show()
pp.imshow( rbp2, interpolation = 'nearest' ); pp.colorbar(); pp.show()

  """

   if shape is None:
      shape = ( sino.shape[ 0 ], sino.shape[ 0 ] )
   img = pr.image( shape )

   if ( sino.top_left[ 1 ] != 1.0 ) or \
      ( sino.bottom_right[ 1 ] != -1.0 ):
         raise ValueError( 'Invalid sinogram t range. Must be ( -1.0, 1.0 )' )
   if ( img.top_left != ( -1.0, 1.0 ) ) or \
      ( img.bottom_right != ( 1.0, -1.0 ) ):
         print img.top_left, img.bottom_right
         raise ValueError( 'Invalid image range. Must be ( -1.0, 1.0 ) x ( -1.0, 1.0 ).' )
   if ( img.shape[ 0 ] != sino.shape[ 0 ] ) or \
      ( img.shape[ 1 ] != sino.shape[ 0 ] ):
         warning.warn( 'Attention: img and sino should preferably have matching dimensions.' )

   # Padded sinogram size (make it even):
   sino_padded_size = int( math.ceil( sino_zp_factor * sino.shape[ 0 ] ) )
   sino_padding_size = sino_padded_size - sino.shape[ 0 ]

   # Fourier radii of FFT irregular samples:
   rhos = np.reshape( np.fft.fftfreq( sino_padded_size ), ( sino_padded_size, 1 ) )
   # Angular positions of irregular samples:
   thetas = np.linspace( sino.top_left[ 0 ], sino.bottom_right[ 0 ], sino.shape[ 1 ] )
   # Trigonometric values:
   trig_vals = np.reshape(
      np.transpose(
         np.array( [ np.sin( thetas ), -np.cos( thetas ) ] )
         ),
      ( 1, thetas.shape[ 0 ] * 2 )
      )
   # Finally, desired irregular samples:
   sample_positions = np.reshape( rhos * trig_vals, ( thetas.shape[ 0 ] * rhos.shape[ 0 ], 2 ) )

   # Computations later required to remove padding.
   delta = sino_padding_size / 2
   extra = sino_padding_size % 2
   odd_sino = sino.shape[ 0 ] % 2

   # Plan nonuniform FFT:
   plan = nf.NFFT(
      d = 2,
      N = img.shape,
      M = sample_positions.shape[ 0 ],
      flags = ( 'PRE_PHI_HUT', 'PRE_PSI' ),
      n = [ i * nfft_zp_factor for i in img.shape  ],
      m = nfft_m
      )
   plan.x = sample_positions
   plan.precompute()

   def fast_radon( img ):
      """
         Compute projection through projection-slice theorem
      """

      # Execute plan:
      plan.f_hat = img
      fsino = plan.trafo()
      #print(fsino.shape)
      # Assemble sinogram:
      fsino = np.reshape( fsino, ( sino_padded_size, sino.shape[ 1 ] ) )
      #print(fsino.shape)
      
      
      # Inverse FFT:
      result = np.fft.ifft( fsino, axis = 0 )
      #print(result.shape)
      
      
      # Shift result:
      result = np.fft.ifftshift( result, axes = ( 0, ) )
      #print(result.shape)
      # Remove padding:
     
      result = result[ delta + odd_sino : result.shape[ 0 ] - delta - extra + odd_sino ]
        
      #print(result.shape)
      # Get real part:
      result = np.real( result )
      
      # Normalize:
      result /= ( 0.5 * ( sino.shape[ 0 ] - 1 ) )
      #Kino = pr.image( result, top_left = (0,1), bottom_right = (math.pi,-1) )
      #pp.imshow(Kino,cmap = 'gray_r', interpolation = 'nearest', 
	#	extent = ( Kino.top_left[ 0 ], Kino.bottom_right[ 0 ], Kino.bottom_right[ 1 ], Kino.top_left[ 1 ] ))
      #pp.show
      # Return image with appropriate bounding box:
      return pr.image( result, top_left = sino.top_left, bottom_right = sino.bottom_right )

   def fast_radon_transpose( sino ):
      """
         Compute backprojection through projection-slice theorem
      """

      # Zero-pad sinogram:
      fsino = np.zeros( ( sino_padded_size, sino.shape[ 1 ] ) )
      fsino[ delta + odd_sino : fsino.shape[ 0 ] - delta - extra + odd_sino ] = sino

      # Shift sinogram columns:
      fsino = np.fft.fftshift( fsino, axes = ( 0, ) )

      # Fourier transform of projections:
      fsino = np.fft.fft( fsino, axis = 0 )

      # Dissasemble transformed sinogram:
      plan.f = np.reshape( fsino, ( fsino.shape[ 0 ] * fsino.shape[ 1 ], 1 ) )

      # Compute adjoint of nouniform Fourier transform:
      result = plan.adjoint()

      # Get real part:
      result = np.real( result )

      # Normalize:
      result /= ( 0.5 * ( sino_padded_size - 1 ) * ( img.shape[ 1 ] - 1 ) )

      return pr.image( result, top_left = img.top_left, bottom_right = img.bottom_right )

   return fast_radon, fast_radon_transpose

if __name__ == '__main__':
   #import pyraft as pr
   #import matplotlib.pyplot as pp
   #import h5py
   #import time

   #f = h5py.File( '/home/elias/curimatã/CM-Day3/tomoTesteH2O.h5', 'r' )
   #v = f[ 'images' ]
   #sino = pr.image( np.transpose( v[ :, 450, : ] ).astype( np.float64 ), x_extent = ( 0.0, math.pi ) )

   #fast_radon, fast_transp = make_fourier_slice_radon_transp( sino )

   #st = time.time()
   #bp = fast_transp( sino )
   #print 'Done!', time.time() - st

   #st = time.time()
   #bp2 = pr.radon_transpose( sino, np.zeros( bp.shape, dtype = np.float64 ) )
   #print 'Done!', time.time() - st

   #print np.max( bp2 ) / np.max( bp ), np.mean( bp2 ) / np.mean( bp )

   #pp.imshow( bp, interpolation = 'nearest' ); pp.colorbar(); pp.show()
   #pp.imshow( bp2, interpolation = 'nearest' ); pp.colorbar(); pp.show()
   #pp.imshow( np.absolute( bp2 - bp ), interpolation = 'nearest' ); pp.colorbar(); pp.show()

   #N = 2048; M = 200
   #img = pr.shepp_logan( ( N, N ) )
   #fast_radon2, fast_transp2 = make_fourier_slice_radon_transp( pr.image( ( N, M ), x_extent = ( 0.0, math.pi ) ) )
   #sino2 = fast_radon2( img )
   #print np.max( sino2 )
   ##pp.imshow( sino2, interpolation = 'nearest', extent = sino2.extent() ); pp.colorbar(); pp.show()
   #sino3 = pr.radon( pr.shepp_logan_desc(), pr.image( ( N, M ), x_extent = ( 0.0, math.pi ) ) )
   ##pp.imshow( sino3, interpolation = 'nearest', extent = sino2.extent() ); pp.colorbar(); pp.show()
   #pp.imshow( np.absolute( ( sino2 - sino3 ) ), extent = sino2.extent(), interpolation = 'nearest' ); pp.colorbar(); pp.show()
   #t = np.arange( sino2.shape[ 0 ] )
   #pp.plot( t, np.absolute( sino2[ :, 0 ] ), t + 1, np.flipud( sino2[ :, sino2.shape[ 1 ] - 1 ] ), 'r' ); pp.show()
   import pyraft as pr
   import matplotlib.pyplot as pp
   import h5py
   import time

   f = h5py.File( '/home/elias/curimatã/CM-Day3/tomoTesteH2O.h5', 'r' )
   v = f[ 'images' ]
   sino = pr.image( np.transpose( v[ :, 450, : ] ).astype( np.float64 ), x_extent = ( 0.0, math.pi ) )

   fast_radon, fast_transp = make_fourier_slice_radon_transp( sino )

   st = time.time()
   bp = fast_transp( sino )
   print 'Done!', time.time() - st

   #st = time.time()
   #bp2 = pr.radon_transpose( sino, np.zeros( bp.shape ) )
   #print 'Done!', time.time() - st

   pp.imshow( bp, interpolation = 'nearest' ); pp.colorbar(); pp.show()
   #pp.imshow( bp2, interpolation = 'nearest' ); pp.colorbar(); pp.show()

   st = time.time()
   rbp = fast_radon( bp )
   print 'Done!', time.time() - st

   #st = time.time()
   #rbp2 = pr.radon( bp, pr.image( np.zeros( sino.shape ), x_extent = ( 0.0, math.pi ) ) )
   #print 'Done!', time.time() - st

   pp.imshow( rbp, interpolation = 'nearest', extent = rbp.extent() ); pp.colorbar(); pp.show()
   #pp.imshow( rbp2, interpolation = 'nearest', extent = rbp2.extent() ); pp.colorbar(); pp.show()
