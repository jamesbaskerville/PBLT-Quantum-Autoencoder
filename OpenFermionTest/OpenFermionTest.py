
# coding: utf-8

# In[8]:


# https://github.com/quantumlib/OpenFermion-Psi4/blob/master/examples/openfermionpsi4_demo.ipynb

from openfermion.hamiltonians import MolecularData
from openfermionpsi4 import run_psi4
import matplotlib.pyplot as plt
from openfermion.transforms import get_fermion_operator, jordan_wigner


# In[55]:


# Set molecule parameters.
basis = 'sto-3g'
multiplicity = 1
bond_length_interval = 0.2
n_points = 10

# Set calculation parameters.
run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 0
run_fci = 1
delete_input = True
delete_output = True

# Generate molecule at different bond lengths.
hf_energies = []
fci_energies = []
bond_lengths = []
hamiltonians = []
n_qubitss = []
for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point)
    bond_lengths += [bond_length]
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    molecule = MolecularData(
        geometry, basis, multiplicity,
        description=str(round(bond_length, 2)),
        filename="{}data".format(basis))
    
    # Run Psi4.
    molecule = run_psi4(molecule,
                        run_scf=run_scf,
                        run_mp2=run_mp2,
                        run_cisd=run_cisd,
                        run_ccsd=run_ccsd,
                        run_fci=run_fci)

    # Print out some results of calculation.
    n_qubitss += [molecule.n_qubits]
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = jordan_wigner(get_fermion_operator(hamiltonian))
    hamiltonians += [hamiltonian]


# In[61]:


from qutip import *
letter_to_op = {
    'X':sigmax(),
    'Y':sigmay(),
    'Z':sigmaz()
}
def makepaulin(info):
    n, l = info
    op = letter_to_op(l)
    tmp = [qeye(2) for _ in range(n)]
    tmp[n] = op
    return tensor(tmp)
        


# In[60]:


import numpy as np
terms = hamiltonians[0].terms
keys = list(terms.keys())
keys


# In[59]:


# create tensored identity in hilbert space of n qubits
def tenseye(n):
    return tensor([qeye(2) for _ in range(n)])


# In[54]:


def ham_to_op(n, of_hamiltonian):
    terms = of_hamiltonian.terms
    
    # generate zeroed operator of correct dimensions
    op = tenseye(n) - tenseye(n)
    
    # iterate through terms of hamiltonian
    for key in list(terms.keys()):
        

