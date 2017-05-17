FISTA readme
To run FISTA with different tomographic operators (NFFT, PPFFT, PFFT) and data sets (Shep-Logan w/o noise, noisy Shep-Logan, Apple seed) do the following:

1) Open FISTA.py
2) Change the value of the varialbe "Method" to either 0, 1, or 2 to use NFFT, PPFFT, or PFFT, respectively
3) Change the value of the variable "Data" to either 0, 1, or 2 to use Clean Shep-Logan, Noisy Shep-Logan, or Apple Seed data, respectively
4) Change the value of the variable "itr" to the desired number of iterations
5) Run FISTA.py