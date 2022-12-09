

# Reverberation Mapping 

You'll need a fortran compiler 

Hopefully you have homebrew installed, which enables us to do 
```bash
brew install gcc
```
so that (as long as gfortran's executable is on the $PATH) we can compile both of 
- zdcf_v2.2.f90
- plike_v4.0.f90

from [here](https://www.weizmann.ac.il/particle/tal/research-activities/software) (Alexander 1997)


by calling

```bash
gfortran zdcf_v2.2.f90 -o zdcf.out
gfortran plike_v4.0.f90 -o plike.out
```
