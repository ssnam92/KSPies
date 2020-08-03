
.. _Userguide:

User guide
==========

.. todo::

    * Need Summary
    * Need to refer to installation 
    * Need WY guide
    * Need ZMP guide
    * Need a slim example of both and the helper.
    * Probably should direct users to our upcoming paper for more scientific details.

Summary
    This is a broad outline on how the software is used. Both examples of WY and ZMP use the helper function. For a scientific discussion and examples on use, please refer to our paper on our software.

Installation
############

For installation please refer to the :ref:`Installation section <Installkspies>`..


Wu and Yang
###########

First run PySCF and extract one-particle density matrix.
A user can 


Zhao-Morrison-Parr
##################

First run PySCF and get one-particle density matrix.

Simple instructions on what to do.


Example usage 
#############

KSPies is PySCF-based program.
KSpies requires at least two input.
1. mole object: which defines geometry, basis set, and other molecular information (charge, spin...) and
2. density matrix: generated with same mole object. Since KSPies is KS inversion program, this will used as a target density.

Below simple example code shows HF and CCSD calculation of Ne atom

.. code-block:: python
  :linenos:

    from pyscf import gto, scf, cc

    mol = gto.M(atom='Ne',basis='cc-pVTZ')
    mf = scf.RHF(mol).run()
    mc = cc.CCSD(mf).run()

And one-particle reduced density matrix (1-rdm) can be obtained from HF and CCSD calculation

.. code-block:: python
  :linenos:

    dm_hf = mf.make_rdm1()
    dm = mc.make_rdm1()
    from kspies import util
    dm_cc = util.mo2ao(mol, dm, mf.mo_coeff)

where the last line convertes mo-basis CCSD 1-rdm to ao-basis, using mo2ao function in util module.

Now, mol and dm_hf (or dm_cc) passes into KSpies to perform inversion.
ZMP can be done as

.. code-block:: python
  :linenos:

    from kspies import zmp
    mz = zmp.RZMP(mol, dm_hf)
    mz.zscf(16)

WY can be done as

.. code-block:: python
  :linenos:

    from kspies import wy
    mw = wy.RWY(mol, dm_hf)
    mw.run()

For detailed settings or outputs, please see ???

Failures
########

Inversion sometimes fails.
Here are some general instructions when inversion fails.

1. Check the dimension of the 1-rdm.
For RWY and RZMP, the 1-rdm should have shape (nao,nao).
for UWY and UZMP, the 1-rdm should have shape (2,nao,nao).

2. Check same mole object is given into PySCF and KSpies.
For example, same spin-state or same number of electrons.

3. Check if the target density is potentially pure-state vs-representable.
If its not, KS solution does not exist.
For example, inversion of C2 electron density will fail (especially when density is generated with multi-reference method) because its not vs-representable.
Please see [Theoretical Chemistry Accounts, 99(5), 329-343] for detail.
Typically, inversion of target density obtained based on ROHF calculation (ROHF, ROHF-UCCSD) does not converge.


Below here shows some instructions when ZMP failes

Increase l gradually with large level shift.
For example,

.. code-block:: python
  :linenos:

    mz = zmp.RZMP(mol, dm_cc) 
    mz.zscf(1024)

will never converge.
However,

.. code-block:: python
  :linenos:

    mz = zmp.RZMP(mol, dm_tar)
    for l in [ 8, 16, 32, 64, 128, 256, 512, 1024]:
        mz.level_shift_factor = l*0.1
        mz.zscf(l)

convergence will be much better.
ZMP, by design, C decreases when l increases.
However, for some large l, integrated density error (dN in the log) may increase when l increases.
This means l is too large for a given basis set, since flexibility of XC potential in ZMP is determined from ao basis.
After this point, SCF convergence will hard.
See [The Journal of Chemical Physics 105, 9200 (1996)] for detail.


Below here shows some instructions when WY fails

Optimization algorithm, BFGS 

