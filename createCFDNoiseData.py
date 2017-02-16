#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:54:52 2017

@author: CupulFamily
"""
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
# mean of distribution
flat_im = np.ones(sino.shape)*(10**4)
flat = flat_im
flat = pr.image( flat , top_left =  (0,1), bottom_right = (math.pi, -1) )
print 'flat size', flat.shape
# Compute counts data based on flat data

counts = flat*np.exp(-sino)
dark = 0*counts
print(type(counts))
print(type(sino))
print(type(flat))
counts = np.random.poisson(counts)
flat = np.random.poisson(flat)
sino = -np.log(counts/flat)

counts = pr.image( counts , top_left =  (0,1), bottom_right = (math.pi, -1) ) 

dark = pr.image( dark , top_left =  (0,1), bottom_right = (math.pi, -1) )
 
flat = pr.image( flat , top_left =  (0,1), bottom_right = (math.pi, -1) )

print 'counts size', counts.shape
print 'flat size',flat.shape
print 'dark size',dark.shape

# pp.imshow( flat, cmap = 'gray_r', interpolation = 'nearest', 
# 	extent = ( flat.top_left[ 0 ], flat.bottom_right[ 0 ], flat.bottom_right[ 1 ], flat.top_left[ 1 ] ) )
# pp.show()
pp.imshow( counts, cmap = 'gray_r', interpolation = 'nearest', 
	extent = ( 0, np.pi , -1, 1))
pp.show()

# Write Shep-Logan phandtom photo to data files
pr.image_write( 'nslcounts.mat', counts, dtype = np.float32)
pr.image_write( 'nslflat.mat', flat, dtype = np.float32)
pr.image_write( 'nsldark.mat', dark, dtype = np.float32)