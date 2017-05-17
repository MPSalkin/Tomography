SSAEM readme
To run SSAEM with different tomographic operators (NFFT, PPFFT) and data sets (Shep-Logan w/o noise, noisy Shep-Logan, Apple seed) do the following:

1) Open SSAEM.py
2) Scroll down to the first comment section under "__main__"
3) Change the value of the varialbe "Method" to either 0 or  1 to use NFFT or PPFFT respectively
4) Change the value of the variable "Data" to either 0, 1, or 2 to use Clean Shep-Logan, Noisy Shep-Logan, or Apple Seed data, respectively
5) Change the value of the variable "M" to the desired number of subsets (only 2 subsets are supported for PPFFT)
6) Change the value of the variable "N" to the desired number of iterations
7) Run SSAEM.py