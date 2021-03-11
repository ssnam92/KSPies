# KSPies
Kohn-Sham Python-based inversion Evaluation Software

Documentaiton: https://ssnam92.github.io/KSPies/index.html
DOI for Code: https://doi.org/10.25351/V3.KS-PIES.2020

This package simplifies evaluating Kohn-Sham inversion potentials and densities for use in the quantum chemistry.
In addition to documentation of the scientific code, this site includes information on installation, setup, use, and examples of use to equip quantum chemists and physicists with this useful toolkit. 
Please let us know if you have further questions, suggestions, or ideas.

Installation
------------
* Prerequisites
    - Python 3.6 or higher
    - Numpy 1.8.0 or higher
    - Scipy 1.4.1 or higher
    - PySCF 1.6.6 or higher
    - LAPACK(for Fortran)

* Code Ocean
  Our code is uploaded in the Code Ocean for verification without dependencies (Link:TODO)

* Installation
  Using pip:

        pip install kspies

  For systems not supported by the manylinux wheel, user can run using the python implimentation, or compile the fortran code `kspies/kspies_fort.f90` for their system.
  Example compile commands are detailed in `kspies/compile.sh`
