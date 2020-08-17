
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

Prerequisites
#############

KSPies is PySCF-based program. PySCF should be installed.
All KSPies inversion program (ZMP and WY) requires at least two input:
1. mole object: which defines geometry, basis set, and other molecular information (charge, spin...) and
2. density matrix: generated with same mole object. Since KSPies is KS inversion program, this will used as a target density.
Below code shows how these inputs can be generated for HF and CCSD calculations.

.. code-block:: python
  :linenos:

    from pyscf import gto, scf, cc

    mol = gto.M(atom='Ne',basis='cc-pVTZ')
    mf = scf.RHF(mol).run()
    dm_hf = mf.make_rdm1()
    mc = cc.CCSD(mf).run()
    from kspies import util
    dm_cc = util.mo2ao(mol, mc.make_rdm1(), mf.mo_coeff)

"mol" is a mole object that contains standard details needed for quantum chemical calculations, 
such as atomic coordinates, number of electrons, and basis sets.
"dm_hf" or "dm_cc" is atomic orbital (ao) representation of one-particle density matrices (1-rdm) from HF or CCSD calculations, respectively.
Basically, PySCF CCSD calculation provides molecular orbital (mo) representation of 1-rdm, 
which can be converted into ao representation by using mo2ao in util.
Any density matrix in ao representation can be feed into inversion program.

Below here shows examples of ZMP and WY.

Zhao-Morrison-Parr
##################

The simplest way of performing ZMP calculation is:

.. code-block:: python
  :linenos:

    from kspies import zmp
    mz = zmp.RZMP(mol, dm_hf)
    mz.zscf(16)

which uses lambda = 16.
Function zscf(l) performs self-consistent ZMP calculations for l.

Wu and Yang
###########

The simplest way of performing WY calculation is:

.. code-block:: python
  :linenos:

    from kspies import wy
    mw = wy.RWY(mol, dm_cc)
    mw.run()

For detailed settings or outputs, please see docs(I don't known how to put link. link to source code docs) 

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
See [The Journal of Chemical Physics 105, 9200 (1996)] for approximate amount of l for given basis set.


Below here shows some instructions when WY fails

If WY "fails", it means scipy.optimize failed to find maximum point of Ws.
Typically, default optimization algorithm 'trust-exact', work generally fine for most of the inversion problems.
However, when potential basis is very large (uncontracted or even-tempered gaussian), 
Hessian is nearly singular and thus Hessian-based optimization algorithms does not work.
In those cases, switch to gradient-based optimization algorithms, BFGS or CG, will might work.
CG typically takes more iteration to converge than BFGS.
However, when BFGS fails, CG can be an option.

If WY fails with maximum gradient element (can be checked with .info() method) approximately 1e-5, 
check if "tol" is set too low.
For molecular systems, setting "tol" below 1e-7 might numerically burden to WY.

Note that the result of WY is very sensitive to initial condition or optimization algorithm used.
For density-rich region (i.e. vicinity of nuclei or bonding region), this is not a problem, 
but for density-deficient region, the shape of potential may depend on optimization conditions.
See [int J Quantum Chem. 2018;118:e25425] for practical details of WY.
