
Examples
========

Included in the KS-pies download is an example folder including several scripts which 
    demonstrate possible uses of KS-pies. Each script intends to showcase the use of 
    a KS-pies feature.


.. _Full-example:

1. Test Script
##############

A complete example showing ZMP, WY and the util functions. 
    A Be atom is used for the test density. 
    Prints to terminal results and expected results, that can
    be compared to confirm that the code is working correctly.


.. _ZMP-benzene:

2 Benzene
#########

Use ZMP to create a KS potential of Benzene.

.. code-block:: python
  :linenos:

    wy.RWY(mol, dm_tar)

Showing use restricted WY instance with **run**, **info**, and use of the stored density matrix **dm**.

.. code-block:: python
  :linenos:

    wy.UWY(mol, udm_tar)

Showing use unrestricted WY instance with **run**, **info**, and use of the stored density matrix **dm**.


.. code-block:: python
  :linenos:

    zmp.RZMP(mol, dm_tar)

Showing use restricted ZMP instance with and without the density fitting procedure (**with_df = True**). Also shows use of **level_shift**, **zscf**, and how various lambda can be used.


.. code-block:: python
  :linenos:

    zmp.UZMP(mol, dm_tar)

Showing use unrestricted ZMP instance with and without the density fitting procedure (**with_df = True**). Also shows use of **level_shift**, **zscf**, and how various lambda can be used.



.. _WY-ZMP-oxygen:

3. Oxygen
#########

Use unrestricted ZMP and unrestricted WY to calculate a KS potential of molecular oxygen.

.. code-block:: python
  :linenos:

    zmp.UZMP(mol, dm_tar)

.. code-block:: python
  :linenos:

    wy.UWY(mol, dm_tar)


.. _WY-regular:

4. Restricted WY
#################

Perform restricted WY on molecular nitrogen.

.. code-block:: python
  :linenos:

    wy.RWY(mol, dm_tar, pbas=PBS)


.. _userdefined-systems:

5. User Defined System
######################

Perform restricted WY on a user defined harmonic oscillator.


wy.RWY(mol,dm_tar,Sijt=Sijt)

To create a user defined instance, a number of settings must be specified:

.. code-block:: python
  :linenos:

    mw = wy.RWY(mol, dm_tar, Sijt=Sijt)
    mw.tol = 1e-7
    mw.method = 'bfgs'
    mw.T = T
    mw.Tp = T
    mw.V = V
    mw.S = S
    mw.guide = None

Which is then executed with:

.. code-block:: python
  :linenos:

    mw.run()
    mw.info()
    mw.time_profile()


.. _plot-xc-zmp:

6. Plot XC ZMP FA
#################

Calculate and plot regularized ZMP using the util functions with exchange-correlation aspects of the Fermi-Amaldi potential.

.. code-block:: python
  :linenos:

    zmp.RZMP(mol, dm_tar)

Showing the **guide**, **level_shift**, **dm**, and **zscf** routines.

.. code-block:: python
  :linenos:

    util.eval_vh(mol, coords, dmxc )

.. _plot-xc-wy:

7. Plot XC WY PBE
#################

Calculate and plot regularized WY using the util functions with a PBE guiding potential.

.. code-block:: python
  :linenos:

    wy.RWY(mol, dm_tar, pbas='cc-pVQZ')

.. code-block:: python
  :linenos:

    util.eval_vxc(mol, dm_tar, mw.guide, coords)
