# This is work in progress, many improvements are necessary ...
## Overall Layout
- In the gui directory, create a config.py file which holds the default path to the directory 
  with lens files and other global settings, such as locale. 

## FFT modulation transfer function (MTF)
- The bin size in the calculation of the line spread function as a histogram 
  must be adjusted based on the diffraction-limited resolution, that is the cut-off frequency,
  to avoid aliasing, which occurs when the resolution of the line spread function is coarser than 
  the test pattern.

## Gaussian beams and Gausslets
- Implement propagation and display of Gaussian beams using ray tracing of two rays

## Chromatic aberrations
- A lens file (JSON) should contain a setup part specifying (fields, wavelengths and aperture)
  in addition to the sequence of surfaces. The sequence of surfaces can be in different configurations.

## Documentation (docstrings for Sphinx)
- **add tests and CI/CD**

## Minor tasks:
- Seidel coefficiens in waves, additional quantities such as paraxial working F-number 
- make n_air adjustable
- Calculate position of nodal points if the index of refraction before and after the lens system is different.



