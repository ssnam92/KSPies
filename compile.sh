#f2py -c --opt='-O4' --f90flags='-fopenmp' -lgomp -llapack kspies_fort.f90 -m kspies_fort
f2py -c --f90flags='-fopenmp' -lgomp -llapack kspies_fort.f90 -m kspies_fort

