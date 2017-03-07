import os
import glob


files = glob.glob('/Users/MPSalkin/Tomography/Visuals/Data/*')
for f in files:
    os.remove(f)

files = glob.glob('/Users/MPSalkin/Tomography/Visuals/Figures/*')
for f in files:
    os.remove(f)

files = glob.glob('/Users/MPSalkin/Tomography/Visuals/Images/*')
for f in files:
    os.remove(f)