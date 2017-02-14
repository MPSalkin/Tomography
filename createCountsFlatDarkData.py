import pyraft as pr
import numpy as np
import fourier_slice as fs
import matplotlib.pyplot as pp
import math
from scipy.stats import poisson as pos

#import shep-logan(phantom photo) 
sl = pr.image_read( 'sl.mat' ) 
print 'sl',sl.shape

sino = pr.image( np.zeros( (1024,1024) ) , top_left =  (0,1), bottom_right = (math.pi, -1) ) 
fast_radon, fast_transp = fs.make_fourier_slice_radon_transp( sino )
sino = fast_radon( sl )

print 'sino size',sino.shape

# Create flat image using Poisson distribution
lamb = 1 # mean of distribution
flat_im = pos.rvs(lamb, size=(sl.shape))
flat = fast_radon(flat_im)

# Compute counts data based on flat data
counts = flat*np.exp(-sino)
dark = 0*counts

print 'counts size', counts.shape
print 'flat size',flat.shape
print 'dark size',dark.shape

# pp.imshow( flat, cmap = 'gray_r', interpolation = 'nearest', 
# 	extent = ( flat.top_left[ 0 ], flat.bottom_right[ 0 ], flat.bottom_right[ 1 ], flat.top_left[ 1 ] ) )
# pp.show()

# pp.imshow( sino, cmap = 'gray_r', interpolation = 'nearest', 
# 	extent = ( sl.top_left[ 0 ], sl.bottom_right[ 0 ], sl.bottom_right[ 1 ], sl.top_left[ 1 ] ))
# pp.show()

# Write Shep-Logan phandtom photo to data files
pr.image_write( 'slcount.mat', counts, dtype = np.float32)
pr.image_write( 'slflat.mat', flat, dtype = np.float32)
pr.image_write( 'sldark.mat', dark, dtype = np.float32)

