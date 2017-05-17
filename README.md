# Tomography
Tomography Algorithms for Math 597


There are two main files where results are produced, FISTA.py and SSAEM.py.
Each of these are run independently but require no input (see the respective readme.txt files for each).

Once the files are run with matching number of iterations, it is possible to visualize differences in the methods by running the file "PlotData.py". The console will prompt for a number of iterations to plot (you can run data with varying iteration numbers for a single data set and all of the data is stored in /Visuals/Data but overwritten upon new run). This will plot three standard curves, Obj Fun vs. Time, Obj Fun Dec. vs. Time, and SSIM vs. Iterations and are stored in the /Visuals/Figures . The best SSIM image is also stored in the /Visuals/Images directory and the filname is created according to which method, how many iterations and how many subsets are used. If you wish to remove certain subsets from which SSAEM was run, you can use "RemoveSubsets.py" and follow the prompts to remove selected subsets from the final plots.

Note that results and images should be pulled out of the directory and placed in a seperate folder prior to running algorithms on a seperate data set, otherwise results are overwritten. 

If you wish to clear the results directories you can run "DeleteVisuals.py" and follow the prompts to clear excess visuals.

NOTE 1: OSTR.py was an early implementation of the OSTR algorithm which was replaced by SSAEM in the final report, however the code does function and can be made useable, however in the interest of time, we, the authors have not made it as easily run as FISTA and SSAEM. Please email the authors for more imformation regarding this and any questions.

NOTE 2: All data used in the reports can be found in /TomoData and new data can easily be added into one of the subfolders directly and easily reconstructed by changing the strings in the importfunctions for a desired data case (note that paramters will not be adjusted automatically, and thus the user must experiment with the values).

NOTE 3: Depedencies include:
  -pyNFFT (which requires other C++ libraries)
  -skimage (scikit-image, for SSIM computation)
  -matplotlib (for all plots)
  -numpy 

-Regards,
Jacob Cupul, Luis Ramirez, Matthew Salkin
