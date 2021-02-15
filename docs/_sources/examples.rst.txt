
Examples
========

Included in the KS-pies download is an example folder including several scripts which 
demonstrate possible uses of KS-pies. Each script intends to showcase the use of 
a KS-pies feature.


.. _Full-example:

1. Test Script
##############

A complete example using ZMP, WY and the util functions. 
A Be atom is used for the test density. 
Prints to terminal results and expected results, that can
be compared to confirm that the code is working correctly.

.. literalinclude:: ../examples/1_test_script.py


.. _ZMP-benzene:

2. Benzene
##########

First perform DF-RHF calculation of benzene,
and then perform RWY, UWY, DF-RZMP, DF-UZMP, RZMP and DF-UZMP calculations with HF target density.
The example also plot density differences (requires matplotlib)
Note that RZMP and UZMP with DF approximation require a substantial amount of time.

.. literalinclude:: ../examples/2_benzene.py


.. _WY-ZMP-oxygen:

3. Oxygen
#########

Similar with benzene example but CCSD density is used as a target

.. literalinclude:: ../examples/3_oxygen.py


.. _WY-regular:

4. Regularized WY
#################

Perform restricted WY on molecular nitrogen with regularization

.. literalinclude:: ../examples/4_regularized_wy.py


.. _userdefined-systems:

5. User Defined System
######################

Perform restricted WY on a user defined harmonic oscillator.
This example shows
generation of finite-difference Hamiltonian, solving HF equation with it,
and perform WY calculation with that HF density as a target 
To create a user defined instance, a number of settings must be specified:

.. literalinclude:: ../examples/5_user_defined_system.py


.. _plot-xc-zmp:

6. Plot XC ZMP FA
#################

Calculate and plot regularized ZMP using the util functions with exchange-correlation aspects of the Fermi-Amaldi potential.
Demonstrating the **guide**, **level_shift**, **dm**, and **zscf** routines.

.. literalinclude:: ../examples/6_plot_xc_zmp_fa.py


.. _plot-xc-wy:

7. Plot XC WY PBE
#################

Calculate and plot regularized WY using the util functions with a PBE guiding potential.

.. literalinclude:: ../examples/7_plot_xc_wy_pbe.py


.. _userdefined-potential:

8. User Defined Potential Basis
###############################

Use of user-defined potential basis.
Slater-type basis is given as an example

.. literalinclude:: ../examples/8_user_defined_potential.py

