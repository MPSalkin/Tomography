#We use the one encoding: utf8
import ctypes
import ctypes.util
import multiprocessing
import math
import fourier_slice as fs
import numpy

nthreads = multiprocessing.cpu_count()

## Load required libraies:
#libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
#libblas   = ctypes.CDLL( ctypes.util.find_library( "blas" ), mode=ctypes.RTLD_GLOBAL )
#libfftw3  = ctypes.CDLL( ctypes.util.find_library( "fftw3" ), mode=ctypes.RTLD_GLOBAL )
#libcairo  = ctypes.CDLL( ctypes.util.find_library( "cairo" ), mode=ctypes.RTLD_GLOBAL )
#libraft   = ctypes.CDLL( ctypes.util.find_library( "raft" ) )

# "double *" type:
_c_double_p = ctypes.POINTER(ctypes.c_double)

class RAFT_MATRIX( ctypes.Structure ):
   """A raft_matrix from raft:"""
   _fields_ = [ ( "p_data", _c_double_p ),
                ( "lines", ctypes.c_int ),
                ( "line_stride", ctypes.c_int ),
                ( "columns", ctypes.c_int ),
                ( "column_stride", ctypes.c_int )
               ]

class RAFT_IMAGE( ctypes.Structure ):
   """A raft_image from raft:"""
   _fields_ = [ ( "data", RAFT_MATRIX ),
                ( "tl_x", ctypes.c_double ),
                ( "tl_y", ctypes.c_double ),
                ( "br_x", ctypes.c_double ),( "br_y", ctypes.c_double )
               ]

## Function prototypes:
#libraft.raft_image_phantom_fromdesc.argtypes = [ RAFT_IMAGE, RAFT_MATRIX ]
#libraft.raft_image_phantom_fromdesc.restype = None
#libraft.raft_radon_fromdesc.argtypes = [ RAFT_IMAGE, RAFT_MATRIX ]
#libraft.raft_radon_fromdesc.restype = None

#libraft.raft_radon_bresenham.argtypes = [ RAFT_IMAGE, RAFT_IMAGE, ctypes.c_int ]
#libraft.raft_radon_bresenham.restype = None
#libraft.raft_backprojection_bresenham.argtypes = [ RAFT_IMAGE, RAFT_IMAGE, ctypes.c_int ]
#libraft.raft_backprojection_bresenham.restype = None
#libraft.raft_radon_transpose_bresenham.argtypes = [ RAFT_IMAGE, RAFT_IMAGE, ctypes.c_int ]
#libraft.raft_radon_transpose_bresenham.restype = None

#libraft.raft_radon_slantstack.argtypes = [ RAFT_IMAGE, RAFT_IMAGE, ctypes.c_int ]
#libraft.raft_radon_slantstack.restype = None
#libraft.raft_backprojection_slantstack.argtypes = [ RAFT_IMAGE, RAFT_IMAGE, ctypes.c_int ]
#libraft.raft_backprojection_slantstack.restype = None

#libraft.raft_haar.argtypes = [ RAFT_MATRIX, ctypes.c_int, ctypes.c_int ]
#libraft.raft_haar.restype = None

#libraft.raft_haar_soft_threshold.argtypes = [ RAFT_MATRIX, ctypes.c_double, ctypes.c_int, ctypes.c_int ]
#libraft.raft_haar_soft_threshold.restype = None

def shepp_logan_desc():
   """
   Description of Shepp-Logan head phantom.

   The collumns of the matrix represent, respectively, intensity, horizontal
   semiaxis, vertical semiaxis, center_x, center_y, rotation angle.
   """
   PIover10 = math.pi / 10.0
   desc = numpy.array( [ [  1.0,  0.69  ,  0.92 ,  0.0 ,  0.0   ,  0.0      ],
                         [ -0.8,  0.6624,  0.874,  0.0 , -0.0184,  0.0      ],
                         [ -0.2,  0.1100,  0.31 ,  0.22,  0.0   , -PIover10 ],
                         [ -0.2,  0.1600,  0.41 , -0.22,  0.0   ,  PIover10 ],
                         [  0.1,  0.2100,  0.25 ,  0.0 ,  0.35  ,  0.0      ],
                         [  0.1,  0.0460,  0.046,  0.0 ,  0.1   ,  0.0      ],
                         [  0.1,  0.0460,  0.046,  0.0 , -0.1   ,  0.0      ],
                         [  0.1,  0.0460,  0.023, -0.08, -0.605 ,  0.0      ],
                         [  0.1,  0.0230,  0.023,  0.0 , -0.606 ,  0.0      ],
                         [  0.1,  0.0230,  0.046,  0.06, -0.605 ,  0.0      ] ]
                     )
   return desc

class image( numpy.ndarray ):
   """This class represents an image. This may not be the ideal foundation, 
   because there are already some options for image classes. Further study is necessary."""

   def __new__(
                subtype,
                shape,
                top_left = None,
                bottom_right = None,
                extent = None,
                x_extent = None,
                y_extent = None,
                **kwargs
              ):
      """Creates and returns a new object of the correct subtype"""

      # Which field-of-view arguments where given?
      extent_given = extent or x_extent or y_extent
      corner_given = top_left or bottom_right

      # Are they acceptable?
      if extent_given and corner_given:
         raise TypeError( 'Mutually exclusive arguments given.' )

      # Extent given, adjust corners:
      if extent_given:
         # Extent given by parts:
         if not extent:
            if not x_extent:
               x_extent = ( -1.0, 1.0 )
            if not y_extent:
               y_extent = ( -1.0, 1.0 )
         # Extent given fully:
         else:
            x_extent = ( extent[ 0 ], extent[ 1 ] )
            y_extent = ( extent[ 2 ], extent[ 3 ] )
         # Finally, we can set up corners:
         top_left     = ( x_extent[ 0 ], y_extent[ 1 ] )
         bottom_right = ( x_extent[ 1 ], y_extent[ 0 ] )

      # pyraft.image given as argument
      if isinstance( shape, image ):

         # Check for given corners:
         if not extent_given:
            if not top_left:
               top_left = shape.top_left
            if not bottom_right:
               bottom_right = shape.bottom_right

         # No arguments other than corners can be taken:
         if kwargs:
            raise TypeError( 'Unhandled arguments!' )

         ## In here, shape is actually a pyraft.image:
         #obj = numpy.asarray( shape ).view( subtype )
         # TODO: No view, make a copy! But there must be a neater way...
         obj = numpy.ndarray.__new__( subtype, shape.shape, **kwargs )
         obj[ ... ] = shape[ ... ]

      else:

         # Check for given corners:
         if not extent_given:
            if not top_left:
               top_left = ( -1.0, 1.0 )
            if not bottom_right:
               bottom_right = ( 1.0, -1.0 )

         # numpy.ndarray given as argument:
         if isinstance( shape, numpy.ndarray ):

            if kwargs:
            # No arguments other than corners can be taken:
               raise TypeError( 'Unhandled arguments!' )

            # In here, shape is actually a numpy.ndarray:
            #obj = numpy.asarray( shape ).view( subtype )
            # TODO: No view, make a copy! But there must be a neater way...
            obj = numpy.ndarray.__new__( subtype, shape.shape, **kwargs )
            obj[ ... ] = shape[ ... ]

         # We must create a zero array:
         else:

            # Default data type is double:
            if not ( 'dtype' in kwargs ):
               kwargs[ 'dtype' ] = numpy.float64
            obj = numpy.ndarray.__new__( subtype, shape, **kwargs )
            obj[ : ] = 0.0

      # All relevant dimensions must match:
      if ( len( obj.shape ) != len( top_left ) ) or ( len( top_left ) != len( bottom_right ) ):
         raise TypeError( 'Dimensions must match!' )

      # Set new attributes:
      obj.top_left = top_left
      obj.bottom_right = bottom_right
      try:
         obj.sampling_distances = ( ( bottom_right[ 0 ] - top_left[ 0 ] ) / ( obj.shape[ 1 ] - 1.0 ),
                                    ( bottom_right[ 1 ] - top_left[ 1 ] ) / ( obj.shape[ 0 ] - 1.0 )
                                  )
      except ZeroDivisionError:
         obj.sampling_distances = ( 0.0, 0.0 )
      return obj

   def __array_finalize__( self, obj ):
      """Set self attributes"""
      if obj is None: return # When ran from __new__

      # Else do the job:
      self.top_left = getattr( obj, 'top_left', None )
      self.bottom_right = getattr( obj, 'bottom_right', None )
      self.sampling_distances = getattr( obj, 'sampling_distances', None )

   def __reduce__( self ):

      # Initial state is only ndarray state:
      full_state = list( numpy.ndarray.__reduce__( self ) )

      #Further attributes:
      image_state = ( self.top_left, self.bottom_right, self.sampling_distances )

      # Add image attributes:
      full_state[ 2 ] = ( full_state[ 2 ], image_state )

      return tuple( full_state )

   def __setstate__( self, state ):

      # Call superclass' __setstate__:
      numpy.ndarray.__setstate__( self, state[ 0 ] )

      # Set our own state:
      self.top_left = state[ 1 ][ 0 ]
      self.bottom_right = state[ 1 ][ 1 ]
      self.sampling_distances = state[ 1 ][ 2 ]

   def sample_coordinates( self, idx ):
      """ Returns coordinates of sample """
      return ( self.top_left[ 0 ] + idx[ 1 ] * self.sampling_distances[ 0 ], self.top_left[ 1 ] + idx[ 0 ] * self.sampling_distances[ 1 ] )

   def get_y_coordinate( self, idx ):
      """ Returns y-coordinate of row """
      return self.top_left[ 1 ] + idx * self.sampling_distances[ 1 ]

   def get_x_coordinate( self, idx ):
      """ Returns x-coordinate of column """
      return self.top_left[ 0 ] + idx * self.sampling_distances[ 0 ]

   # Extent:
   def extent( self ):
      return ( self.top_left[ 0 ], self.bottom_right[ 0 ], self.bottom_right[ 1 ], self.top_left[ 1 ] )

def make_RAFT_MATRIX( array ):
   """Mak a raft_matrix from a numpy.ndarray"""
   return RAFT_MATRIX( ctypes.cast( array.ctypes.data, _c_double_p ), 
                       ctypes.c_int( array.shape[ 0 ] ),
                       ctypes.c_int( array.strides[ 1 ] / 8 ),
                       ctypes.c_int( array.shape[ 1 ] ),
                       ctypes.c_int( array.strides[ 0 ] / 8 )
                      )

def make_RAFT_IMAGE( array, top_left = ( -1.0, 1.0 ), bottom_right = ( 1.0, -1.0 ) ):
   """Make a raft_matrix from a numpy.ndarray from a pyraft.image or from a pyraft.RAFT_MATRIX"""
   if isinstance( array, numpy.ndarray ):
      return RAFT_IMAGE( make_RAFT_MATRIX( array ),
                         ctypes.c_double( top_left[ 0 ] ),
                         ctypes.c_double( top_left[ 1 ] ),
                         ctypes.c_double( bottom_right[ 0 ] ),
                         ctypes.c_double( bottom_right[ 1 ] )
                       )
   elif isinstance( array, RAFT_MATRIX ):
      return RAFT_IMAGE( array,
                         ctypes.c_double( top_left[ 0 ] ),
                         ctypes.c_double( top_left[ 1 ] ),
                         ctypes.c_double( bottom_right[ 0 ] ),
                         ctypes.c_double( bottom_right[ 1 ] )
                       )
   elif isinstance( array, image ):
      return RAFT_IMAGE( array,
                         ctypes.c_double( array.top_left[ 0 ] ),
                         ctypes.c_double( array.top_left[ 1 ] ),
                         ctypes.c_double( array.bottom_right[ 0 ] ),
                         ctypes.c_double( array.bottom_right[ 1 ] )
                       )

#def shepp_logan( shape = ( 256, 256 ), **kwargs ):
   #"""Creates an image with samples from the Shepp-Logan head phantom"""

   ## Phantom description:
   #desc = shepp_logan_desc()

   ## Extract C-struct from numpy.array:
   #DESC = make_RAFT_MATRIX( desc )

   ## Create pyraft.image:
   #img = image( shape, **kwargs )

   ## Extract C-struct from pyraft.image:
   #IMG = make_RAFT_IMAGE( img, img.top_left, img.bottom_right )

   ## Call libraft function!
   #libraft.raft_image_phantom_fromdesc( IMG, DESC );

   ## Return image:
   #return img

#def phantom( desc = shepp_logan_desc, shape = ( 256, 256 ), **kwargs ):
   #"""
   #Creates an image with samples from the described phantom.
   #"""

   ## Extract C-struct from numpy.array:
   #DESC = make_RAFT_MATRIX( desc )

   ## Create pyraft.image:
   #img = image( shape, **kwargs )

   ## Extract C-struct from pyraft.image:
   #IMG = make_RAFT_IMAGE( img, img.top_left, img.bottom_right )

   ## Call libraft function!
   #libraft.raft_image_phantom_fromdesc( IMG, DESC );

   ## Return image:
   #return img

class radon_method:
   """Methods for projection/backprojection operations."""

   BRESENHAM = 1,
   SLANTSTACK = 2,
   HIERARCHICAL = 3,
   FOURIER_SLICE = 4

def radon( img, shape, method = radon_method.BRESENHAM, **kwargs ):
   """Computes the Radon transform using one of several methods"""

   # If not an image, test for Radon-space extents:
   if ( not isinstance( shape, image ) ) and \
      ( not ( 'top_left' in kwargs ) ) and \
      ( not ( 'bottom_right' in kwargs ) ) and \
      ( not ( 'x_extent' in kwargs ) ) and \
      ( not ( 'y_extent' in kwargs ) ):
      if 'theta_extent' in kwargs:
         kwargs[ 'x_extent' ] = kwargs[ 'theta_extent' ]
         del kwargs[ 'theta_extent' ]
      else:
         kwargs[ 'x_extent' ] = ( 0.0, math.pi )
      if 't_extent' in kwargs:
         kwargs[ 'y_extent' ] = kwargs[ 't_extent' ]
         del kwargs[ 't_extent' ]
   # Create pyraft.img to hold sinogram:
   sino = image( shape, **kwargs )
   SINO = make_RAFT_IMAGE( sino, sino.top_left, sino.bottom_right )

   # img is a pyraft.image: compute discrete Radon transform:
   if isinstance( img, image ):
      IMG = make_RAFT_IMAGE( img, img.top_left, img.bottom_right )
      if method == radon_method.BRESENHAM:
         libraft.raft_radon_bresenham( IMG, SINO, ctypes.c_int( nthreads ) )
      elif method == radon_method.SLANTSTACK:
         libraft.raft_radon_slantstack( IMG, SINO, ctypes.c_int( nthreads ) )
      else:
         raise TypeError( 'Unsupported method for Radon Transform!' )
      return sino

   # img is a numpy.array: consider it a description and compute
   # exact Radon transform:
   elif isinstance( img, numpy.ndarray ):
      DESC = make_RAFT_MATRIX( img )
      libraft.raft_radon_fromdesc( SINO, DESC )
      return sino

def backprojection( sino, shape, method = radon_method.BRESENHAM, **kwargs ):
   """Computes the Backprojection using one of several methods"""

   # Create pyraft.img to hold backprojection:
   img = image( shape, **kwargs )
   IMG = make_RAFT_IMAGE( img, img.top_left, img.bottom_right )

   # Compute discrete Backprojection:
   SINO = make_RAFT_IMAGE( sino, sino.top_left, sino.bottom_right )
   if method == radon_method.BRESENHAM:
      libraft.raft_backprojection_bresenham( SINO, IMG, ctypes.c_int( nthreads ) )
   elif method == radon_method.SLANTSTACK:
      libraft.raft_backprojection_slantstack( SINO, IMG, ctypes.c_int( nthreads ) )
   else:
      raise TypeError( 'Unsupported method for Backprojection!' )
   return img

def radon_transpose( sino, shape = None, method = radon_method.BRESENHAM, **kwargs ):
   """Computes the transpose of discrete Radon Transform using one of several methods"""

   # No shape given, use given sinogram dimensions:
   if shape is None:
      shape = ( sino.shape[ 0 ], sino.shape[ 0 ] )

   # Create pyraft.img to hold backprojection:
   img = image( shape, **kwargs )
   IMG = make_RAFT_IMAGE( img, img.top_left, img.bottom_right )

   # Compute transpose:
   SINO = make_RAFT_IMAGE( sino, sino.top_left, sino.bottom_right )
   if method == radon_method.BRESENHAM:
      libraft.raft_radon_transpose_bresenham( SINO, IMG, ctypes.c_int( nthreads ) )
   else:
      raise TypeError( 'Unsupported method for transpose Radon!' )
   return img


def make_fourier_em( sino, shape = None, weight = None, **kwargs ):
   """
      Create a fast em algorithm. Is called by pyraft.make_em()
      when passed
         projection_method = FOURIER_SLICE
   """

   # Default shape depends on sinogram spatial resolution:
   if shape is None:
      shape = ( sino.shape[ 0 ], sino.shape[ 0 ] )

   # Create transform functions:
   radon, radon_transpose = fs.make_fourier_slice_radon_transp( sino, shape, **kwargs )

   # Check if we have to compute weight:
   if weight is None:

      # Weight will be backprojection of ones:
      one = image(
         numpy.ones( sino.shape ),
         top_left = sino.top_left,
         bottom_right = sino.bottom_right
      )

      # Compute weight:
      weight = radon_transpose( one )

      del one

   # EM algorithm:
   def em( x ):
      # Compute Radon transform:
      Ax = radon( x )

      # Iteration:
      Ax = sino / Ax
      rt = radon_transpose( Ax )

      x *= rt
      x /= weight

      return x

   return em


def make_em( sino,
             shape = None,
             projection_method = radon_method.BRESENHAM,
             backprojection_method = None,
             weight = None,
             **kwargs ):
   """
   make_em( sino,
            shape,
            projection_method = radon_method.BRESENHAM,
            backprojection_method = None,
            weight = None,
            **kwargs
          )

   This function returns a function that performs iterations of the EM algorithm
   for tomograpohic image reconstruction. The arguments are as follows:

      sino  : a pyraft.image containing the tomgoraphic data;
      shape : Shape of the reconstructed image, or a numpy.ndarray or a pyraft.image.
              This argument determinates the size and field-of-view of the reconstruction;
      projection_method     : Method to be used for projection;
      backprojection_method : Method to be used for backprojection;

      Further arguments are accepted only when a pyraft.image is not given and are
      forwarded to pyraft.image() constructor, which is called in order to construct
      an weight image required by the algorihtm. An exception to this rule is when
         projection_method == radon_method.FOURIER_SLICE
      In this case, extra arguments will be handled to
         fourier_slice.make_fourier_slice_radon_transp
      Read documentation of that function in order to understand how to gain control
      over parameters for the non-uniform FFT algorithm used by the Fourier method.

      Usage example:

import pyraft
import math
import numpy
import matplotlib.pyplot

# Radon transform of the discrete Shepp-Logan phantom.
# The phantom is 512 x 512, while the sinogram has 600 views sample at 512 points each:
sino = pyraft.radon( pyraft.shepp_logan( ( 512, 512 ) ), ( 512, 600 ), x_extent = ( 0.0, math.pi ) )
matplotlib.pyplot.imshow( sino, extent = sino.extent() ); matplotlib.pyplot.show()

# Starting image for iteration:
x = pyraft.image( numpy.ones( ( 512, 512 ) ) )

# Creates EM function:
em = pyraft.make_em( sino, x )
#Ten EM iterations:
for i in range( 0, 10 ):
   x = em( x )

matplotlib.pyplot.imshow( x, extent = x.extent() ); matplotlib.pyplot.show()
   """
   if shape is None:
      shape = ( sino.shape[ 0 ], sino.shape[ 0 ] )

   if backprojection_method is None:
      backprojection_method = projection_method

   # Method is fourier_slice:
   if ( projection_method     == radon_method.FOURIER_SLICE ) or \
      ( backprojection_method == radon_method.FOURIER_SLICE ):
      if ( projection_method     != radon_method.FOURIER_SLICE ) or \
         ( backprojection_method != radon_method.FOURIER_SLICE ):
            raise ValueError( 'Projection/backprojection methods must be the same for FOURIER_SLICE.' )
      return make_fourier_em( sino, shape, weight, **kwargs )


   # We have to compute weight:
   if not weight:
      one = image(
         numpy.ones( sino.shape ),
         top_left = sino.top_left,
         bottom_right = sino.bottom_right
      )

      # If shape argument is an image, we need to create
      # a weight with same geometry:
      if isinstance( shape, image ):
         weight = radon_transpose(
            one,
            shape.shape,
            top_left = shape.top_left,
            bottom_right = shape.bottom_right,
            method = backprojection_method,
            **kwargs
         )
      # Otherwise we use the provided arguments:
      else:
         weight = radon_transpose( one, shape, method = backprojection_method, **kwargs )

   # EM algorithm:
   def em( x ):
      # Compute Radon transform:
      Ax = radon(
         x,
         sino.shape,
         top_left = sino.top_left,
         bottom_right = sino.bottom_right,
         method = projection_method
      )
      # Iteration:
      Ax = sino / Ax
      x = x * radon_transpose(
         Ax,
         x.shape,
         top_left = x.top_left,
         bottom_right = x.bottom_right,
         method = backprojection_method
      ) / weight

      return x

   return em

def em(
   file_name,
   volume_slice,
   shape = None,
   volume_name = 'images',
   niter = 1,
   t_extent = ( -1.0, 1.0 ),
   theta_extent = ( 0.0, math.pi ),
   **kwargs
   ):

   import h5py

   # Read sinogram:
   f = h5py.File( file_name, 'r' )
   vol = f[ volume_name ]
   sino = image(
      numpy.transpose( vol[ :, volume_slice, : ] ).astype( 'float64' ),
      x_extent = theta_extent,
      y_extent = t_extent
   )

   if shape == None:
   # reconstruction size according to data resolution,
   # ignoring angular sampling rate:
      shape = vol[ 0, :, : ].shape

   if isinstance( shape, image ):
   # Initial image was provided:
      if kwargs:
         raise TypeError( 'Unhandled arguments!' )
      x = shape
   else:
   # Use uniform starting image:
      x = image( shape, **kwargs )
      x[:] = 1.0

   algo = make_em( sino, x )
   for i in range( 0, niter ): x = algo( x )

   return x

def haar( array, nlevels = None ):

   if nlevels is None:
      nlevels = int( math.floor( math.log( min( array.shape ), 2 ) ) )

   if isinstance( array, image ):
      retval = image(
         array.copy(),
         top_left = array.top_left,
         bottom_right = array.bottom_right
         )
   else:
      retval = array.copy()

   M = make_RAFT_MATRIX( retval )

   libraft.raft_haar( M, ctypes.c_int( nlevels ), ctypes.c_int( 1 ) )

   return retval

def hst( array, threshold, n_levels = None ):

   if n_levels is None:
      n_levels = int( math.floor( math.log( min( array.shape ), 2 ) ) )

   if isinstance( array, image ):
      retval = image(
         array.copy(),
         top_left = array.top_left,
         bottom_right = array.bottom_right
         )
   else:
      retval = array.copy()

   M = make_RAFT_MATRIX( retval )

   libraft.raft_haar_soft_threshold( M, ctypes.c_double( threshold ), ctypes.c_int( n_levels ), ctypes.c_int( 1 ) )

   return retval

if __name__ == "__main__":

   import matplotlib.pyplot as pp

   im = shepp_logan( ( 1024, 1024 ) )
   hstim = haar( im )
   pp.imshow( hstim, extent = hstim.extent(), interpolation = 'nearest', vmax = 1, vmin = -1 ); pp.colorbar(); pp.show()

def image_read( filename, dtype = numpy.float64, stype = numpy.int32, order = 'f' ):

   with open( filename, 'r' ) as f:
      shape = numpy.fromfile( f, stype, 2 )
      data = numpy.fromfile( f, dtype, shape[ 0 ] * shape[ 1 ] ).reshape( shape, order = order )
      tl = numpy.fromfile( f, dtype, 2 )
      br = numpy.fromfile( f, dtype, 2 )
      return image( data, top_left = ( tl[ 0 ], tl[ 1 ] ), bottom_right = ( br[ 0 ], br[ 1 ] ) )

def image_write( filename, img, dtype = numpy.float64, stype = numpy.int32, order = 'f' ):

   with open( filename, 'w' ) as f:
      shape = numpy.array( img.shape, dtype = stype )
      shape.tofile( f )
      if order == 'c':
         img.astype( dtype ).tofile( f )
      else:
         img.astype( dtype ).T.tofile( f )
      bbox = numpy.array( ( img.top_left[ 0 ], img.top_left[ 1 ], img.bottom_right[ 0 ], img.bottom_right[ 1 ] ), dtype = dtype )
      bbox.tofile( f )
