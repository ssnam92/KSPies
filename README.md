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
    - opt\_einsum 3.2.0 or higher
    - LAPACK

* Installation\\
  Using pip:

        pip install kspies

  All prerequisite packages are installed automatically except pyscf\
  Or user can download source from GitHub repo, and manually compile `kspies/kspies_fort.f90`.
  Example compile commands can be found in `kspies/compile.sh`

Citing KSPies
-------------
[KS-pies: Kohn–Sham inversion toolkit](https://aip.scitation.org/doi/10.1063/5.0040941),
S. Nam, R. J. McCarty, H. Park, E. Sim, J. Chem. Phys. 154, 124122 (2021); https://doi.org/10.1063/5.0040941
