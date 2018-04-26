
# coding: utf-8

# In[57]:


import numpy as np
from qutip import *
from scipy.optimize import minimize
import h5py
import matplotlib.pyplot as plt


# #### Parameters
# 
# For n qubits, we need n(n-1) + 2n arbitary rotation gates, with 3 parameters each.
# 
# https://arxiv.org/abs/1612.02806

# ### QuTip Implementation

# #### QuTip Helper Functions

# In[2]:


# apply given single-qubit gate to any qubit in system of n qubits
def tgtgate(n, gate, tgt):
    lst = [qeye(2) for _ in range(n)]
    lst[tgt] = gate
    return tensor(lst)

# create tensored identity in hilbert space of n qubits
def tenseye(n):
    return tensor([qeye(2) for _ in range(n)])

# create operator from product of all gates in a circuit
def gate_prod(n, gates):
    prod = tenseye(n)
    for gate in gates:
        prod = prod * gate
    return prod


# #### Arbitrary Rotation Gates (single-qubit and controlled)

# In[3]:


# https://arxiv.org/abs/quant-ph/9503016

# ROT = Rz(alpha) * Ry(theta) * Rz(beta)
def rot(n, params, tgt):
    alpha, theta, beta = params
    return rz(alpha, n, tgt)*ry(theta, n, tgt)*rz(beta, n, tgt)

# CROT = A * CNOT * B * CNOT * C
# A = Rz(alpha)*Ry(theta / 2)
# B = Ry(-theta / 2) * Rz(-(alpha + beta) / 2)
# C = Rz((beta - alpha) / 2)
def ctrl_rot(n, params, ctrl, tgt):
    alpha, theta, beta = params
    A = rz(alpha, n, tgt) * ry(theta / 2.0, n, tgt)
    B = ry(-theta / 2.0, n, tgt) * rz(-(alpha + beta) / 2.0, n, tgt)
    C = rz((beta - alpha) / 2.0, n, tgt)
    assert (A*B*C == tenseye(n))
    assert (A*tgtgate(n, sigmax(), tgt)*B*tgtgate(n, sigmax(), tgt)*C == rot(n, params, tgt))
    return A * cnot(n, ctrl, tgt) * B * cnot(n, ctrl, tgt) * C


# #### Parameter Manipulation

# In[4]:


def init_params(n_params, method=np.ones):
    return method(n_params)

def split_params(n, params):
    return (params[:3*n].reshape(n, 3),
            params[3*n:-3*n].reshape(n, n-1, 3),
            params[-3*n:].reshape(n, 3))

def recombine_params(first, mid, last):
    return np.concatenate((first.flatten(), mid.flatten(), last.flatten()))


# #### Circuit
# The circuit outlined in red below is the unitary gate for encoding (in this case, for 4 qubit inputs).
# ![arbitrary_rotation_gate_circuit](https://image.ibb.co/ji9XBc/unit_cell_arb_rot.png)

# In[5]:


# # create circuit from parameters
# def get_circuit_gates(num_gates, params):
#     gates = np.array([tenseye(4) for _ in range(num_gates)])

#     # R1 - R4
#     for gate_num in range(0, 4):
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = arb_rot(n, alpha, theta, beta, gate_num - 0)

#     # R5 - R7 controlled by qubit 0
#     for gate_num in range(4, 7):
#         control = 0
#         targets = [1, 2, 3]
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = ctrl_rot(n, alpha, theta, beta, control, targets[gate_num - 4])

#     # R8 - R10 controlled by qubit 1
#     for gate_num in range(7, 10):
#         control = 1
#         targets = [0, 2, 3]
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = ctrl_rot(n, alpha, theta, beta, control, targets[gate_num - 7])

#     # R11 - R13 controlled by qubit 2
#     for gate_num in range(10, 13):
#         control = 2
#         targets = [0, 1, 3]
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = ctrl_rot(n, alpha, theta, beta, control, targets[gate_num - 10])

#     # R14 - R16 controlled by qubit 3
#     for gate_num in range(13, 16):
#         control = 3
#         targets = [0, 1, 2]
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = ctrl_rot(n, alpha, theta, beta, control, targets[gate_num - 13])

#     # R17 - R20
#     for gate_num in range(16, 20):
#         alpha, theta, beta = params[gate_num]
#         gates[gate_num] = arb_rot(n, alpha, theta, beta, gate_num - 16)

#     return gates


# In[6]:


# gate_product = gate_prod(n, gates)
# gate_product


# #### Circuit Implementation

# In[7]:


# The n rotation gates (one on each qubit) that happen at the start and end.
def wrapper(n, params):
    assert (len(params) == n)
    gates = []
    for tgt, rot_params in enumerate(params):
        gates.append(rot(n, rot_params, tgt))
    return gate_prod(n, gates)

# The n-1 gates in each "blue box" (set of controlled rotation gates)
def blue_box(n, params, ctrl):
    p_index = 0
    gates = []
    for tgt in range(n):
        #print (tgt, ctrl)
        if tgt == ctrl:
            continue
        rot_params = params[p_index]
        p_index += 1
        gates.append(ctrl_rot(n, rot_params, ctrl, tgt))
    return gate_prod(n, gates)

# The circuit as a whole -- front wrapper, blue boxes, back wrapper
def create_circuit(n, all_params):
    gates = []
    
    # split parameters
    f, m, b = split_params(n, all_params)
    
    # front wrapper
    gates.append(wrapper(n, f))
    
    # blue boxes
    for i in range(n):
        gates.append(blue_box(n, m[i], i))
    
    # back wrapper
    gates.append(wrapper(n, b))
    
    return gate_prod(n, gates)


# In[8]:


# returns n, number of gates needed, number of params needed
def init_consts(n):
    n_gates = n * (n - 1) + 2 * n
    n_params = 3 * n_gates
    return n, n_gates, n_params

# given input state and output state, returns estimated fidelity
#     we can cast to integer because this is the norm squared
#     there's no longer any complex component
def overlap(inp, oup):
    ol = inp.overlap(oup)
    return int(ol * ol.conj())
v_overlap = np.vectorize(overlap)

# return objective to minimize for scipy optimizers
#     given N data points
#     params: parameters to tune
#     args: [n, ...bunch_of_instates_to_train_on...]
def obj_func(params, *args):
    n = args[0]
    in_states = args[1:]
    
    # create encoding operator from parameters of the rotation gates
    encoding_op = create_circuit(n, params)
    
    # apply encoding circuit to all training data
    # (should probably split this up into epochs)
    out_states = encoding_op * in_states
    
    overlaps = 1 - v_overlap(in_states, out_states)
    return sum(overlaps)


# In[9]:


# create qubit from a rand float
def qubit(a, b):    
    # random phase shifts
    if np.random.rand() <= 0.5:
        a = a * 1.0j
    if np.random.rand() <= 0.5:
        b = b * 1.0j
    return Qobj([[a],[b]]).unit()
v_qubit = np.vectorize(qubit)

def gen_data(n_orig, n_enc, data_count=100):
    assert(n_orig >= n_enc)
    data = []
    
    # choose which qubits to exclude
    excluded = np.random.choice(range(n_orig), size=(n_orig - n_enc), replace=False)
    
    # set shape matrix
    shape = [2]*n_orig
    
    # generate data
    for _ in range(data_count):
        qubits = v_qubit(2 * np.random.random(n_orig) - 1, 2 * np.random.random(n_orig) - 1)
        qubits[excluded] = basis(2,0)
        data.append((tensor(qubits)).unit())
        
    return np.array(data)


# In[10]:


# data = gen_data(2,1,data_count=1000)
# initial_params = init_params(num_params)
# data[:5]
# # minimize(obj_func, initial_params, method='Nelder-Mead')


# In[11]:


# U = sigmax()
# instates = np.array([Qobj([[0],[1]]), Qobj([[1],[0]])])
# outstates = U*instates
# sum(1-v_overlap(instates, outstates))


# ## Density Matrix Overview:
# 
# In order to understand how the cost function is computed for the autoencoder, we will need to describe our system in terms of density matricies. Density matrices are useful because they describe ensemble behavior of quantum experiments - which are often the result of imperfect quantum manipulations.
# 
# To this point in ES170 we have dealt only with pure states - where the quantum state of the system inputs and outputs can be described by a single state vector.
# 
# In practice we can have states which are a probablilistic ensemble of pure states. Note that this is fundamentally different from a superposition. We will explore describe this distinction explicitly below. Density matrices are a useful tool for describing this more general class of ensemble states.
# 
# The following examples will illustrate the mechanics of density matrices.
# 
# For a pure state, $|\psi>$ the density matrix (sometimes called the density operator) is given by the following:
# 
# $\rho = |\psi><\psi|$
# 
# For a single bit system, expressed in the $\{|0>,|1>\}$ basis, the density matrix elements are thus:
# 
# $\rho_{11} = <0|\rho|0>$
# 
# $\rho_{12} = <0|\rho|1>$
# 
# $\rho_{21} = <1|\rho|0>$
# 
# $\rho_{12} = <1|\rho|1>$
# 
# With some simple dirac manipulation, we can see that the entries of the density matrix are probabilities. For example, $\rho_{11}$ is the probability of measuring $|\psi>$ in state zero.
# 
# As an example, we construct the density matrix for basis states:

# In[12]:


# Basis state 0 
q0 = basis(2,0)

rho = q0*q0.dag()

rho


# In[13]:


# note the following function also works
ket2dm(q0)


# In[14]:


# For basis state 1
q1 = basis(2,1)

ket2dm(q1)


# These pure, orthogonal basis states are equivalent to fock (photon number) states, and qutip has a built in function:

# In[15]:


fock_dm(2,1)


# Nice! We can simply read the probability of the system being in state zero or one by looking at the diagonal entries. What happens for a superpostion?

# In[16]:


# equal superposition state
q = 1/np.sqrt(2)*(basis(2,0)+basis(2,1))

ket2dm(q)


# For a superposition we have off diagonal terms! These off-diagonal terms have important physical meaning (in some cases can be interpreted as coherence). For now we'll ignore that.
# 
# The most important property is that for a pure state, the trace (sum of diagonals) of the density matrix is 1. This is related exactly to the normalized nature of a pure state.

# In[17]:


ket2dm(q).tr()


# Now let's consider a statistical ensemble of pure states. For example, imagine we prepare a _set_ of qubits, half in state zero and half in state one. We could measure these qubits in sequence and build up a statistical interpretation of the initial state. 
# 
# This statistical outcome can be expressed as an ensemble state:
# 
# $|\Psi> = \frac{1}{2}|0> + \frac{1}{2}|1>$
# 
# For which the density matrix is defined:
# 
# $\rho = \sum\limits_{i}{p_i|\psi_i><\psi_i|}$

# In[18]:


q = 1/2*basis(2,0) + 1/2*basis(2,1)

ket2dm(q)


# ### Probabilities of Pure and Mixed States:
# How can we tell this is not a pure state? By looking at the trace of the density matrix. 
# 
# $\text{Tr}[\rho] < 1 \implies \text{Mixed State}$
# 
# 
# The trace in this case is less than 1, indicating a statistical mixture. Actually, in this case we have what is a maximally mixed state:
# 
# $\text{Tr}[\rho_{max}] = 1/D$ where $D$ is the dimensionality of the system. In the case of qubits D = 2.

# In[19]:


ket2dm(q).tr()


# ### Transformation of Density Matrices:
# 
# Note that density matrices undergo unitary transformation as follows:
# 
# $\rho = |\psi><\psi|$
# 
# $\rho' = |\psi'><\psi'|$
# 
# $|\psi'> = U|\psi>$ and $ <\psi'| =  <\psi|U^{\dagger}$
# 
# $ \implies \rho' = U|\psi><\psi|U^{\dagger}$
# 
# $ \implies \rho' = U\rho U^{\dagger}$

# ## The Cost Fuction:
# 
# Consider an input ensemble state $\{p_i,|\psi_i\rangle\}$, where $p_i$ are ensemble probabilities of each pure state, $|\psi_i\rangle$. Note that ${|\psi_i\rangle}$ are  not necessarily orthonormal. The density matrix is given by the following:
# 
# $\rho = \sum\limits_{i}{p_i|\psi_i\rangle\langle\psi_i|}$
# 
# e.g. if our state turns out to be pure for a pure state (e.g. $|\psi_0\rangle$), the ensemble set reduces to $\{1,|\psi_0\rangle\}$.
# 
# The autoencoder cost function is given by:
# 
# $C_2(q) = \sum\limits_{i}{p_i \times F(Tr_{A}[U^{q}|\psi_i\rangle\langle\psi_i|_{AB}(U^{q})^{\dagger}],|a\rangle_{B})}$
# 
# $C_2(q) = {F(Tr_{A}[U^{q}\rho_{AB}(U^{q})^{\dagger}],\rho_{out,B})}$
# 
# where $q$ is the set of parameters used to form the arbitrary transformation matrix $U^q$, and $\rho_{out,B}$ our trash state. Performing the partial trace over A, reduces the n+k dimensional density matrix to a k dimensional density matrix corresponding only to the trash state subspace.
# 
# Here is a translation of the math into code: 
# * Input state can be, in general, an ensemble state decomposed as $\{p_i,|\psi_i\rangle\}$
# * Represent this set of states as a density matrix
# * Apply the autoencoder transform U to the state, with parameters q
# * Take a partial trace over the trash subset of the transformed density matrix to recover the trash state density matrix
# * Using a fidelity function, compare the trash state density matrix to the ancillary state density matrix (in our case zeros)
# 
# We can take the output of the cost function and feedback into the minimizer to find the encoding transformation.

# Qutip Partial Trace: http://qutip.org/docs/3.1.0/guide/guide-tensor.html
# * example of partial trace function
# * ket to density matrix function
# 
# Qutip Fidelity Function: http://qutip.org/docs/3.1.0/apidoc/functions.html
# * Search fidelity - computes the fidelity of two density matrices

# In[20]:


# Cost Function Testing:
p = [0.5,0.5]
pure = [1,1]
psi = [basis(2,0),basis(2,1)]
psi_pure = 1/np.sqrt(2)*(basis(2,0)+basis(2,1))
psi_mixed = 1/2*(basis(2,0)+basis(2,1))

C1 = np.sum(fidelity(ket2dm(psi_pure),ket2dm(psi_pure)))
print(C1)

C1 = [p[i]*fidelity(ket2dm(psi[i]),ket2dm(psi_mixed)) for i in range(len(psi))]
print(C1)

print(1/2*ket2dm(psi[0])+1/2*ket2dm(psi[1]))

print(ket2dm(1/2*psi[0]+1/2*psi[1]))


# In[25]:


# Cost function implementation from the paper:
n = 5
k = 2

trashdm = ket2dm(tensor([basis(2,0) for _ in range(k)]))

# This version is broken down in terms of basis vectors and ensemble probabilities:
#C2 = np.sum([p_set[i] * fidelity((U*ket2dm(psi_set[i])*U.dag()).ptrace(np.arange(n,n+k)),trashdm) for i in range(len(psi_set))])

# # This version composes the input density matrix from the ensemble set of input states, then performs the transform:
# inputdm = np.sum([p_set[i] * ket2dm(psi_set[i]) for i in range(len(psi_set))])
# C2 = fidelity((U*inputdm*U.dag()).ptrace(np.arange(n,n+k)),trashdm)


# ## Hydrogen Wavefunction Training Set:
# 
# In the infamous article, the authors test the autoencoder with a set of ground state wavefunctions of molecular hydrogen. The autoencoder is trained with a subset of 6 ground state wavefunctions at different radial H-H distances, and then the encoder is tested on states at other radii. The Hamiltonian for molecular hydrogen can be solved numerically (classically), which would give us a benchmark for the performance of the autoencoder. We are considering to also implement this set of test states - though it is proving to be challenging!
# 
# The Hamiltonian for molecular hydrogen, under the Born-Oppenheimer non-relativistic approximation is given by:
# 
# $H = h_{nuc} +\sum\limits_{pq}h_{pq}a_p^{\dagger}a_q^{\dagger} + \frac{1}{2}\sum\limits_{pqrs}h_{pqrs}a_p^{\dagger}a_q^{\dagger}a_r a_s$
# 
# There are a few terms here: a nuclear term $h_{nuc}$, a single electron term, and a two electron term, with corresponding electron integrals $h_{pq}$ and $h_{pqrs}$. These terms express the probabilities of electrons occupying different molecular orbitals (e.g. $a_p^{\dagger}$ creates an electron in spin-orbital $p$).
# 
# As in the case of qubits, we are free to choose the basis of this Hamiltonian to be whatever we like. In quantum chemistry, we choose a basis which corresponds to a specific set of orthogonal electron orbitals. Often, this basis is selected with the convenience of calculation in mind. For example, the STO-6G hydrogen minimal basis set is one such basis set which is optimized for numerical approximation (the details of which we will perhaps need to explore later).
# 
# Furthermore, in order to map this Hamiltonian to a quantum computer, we need to decompose it as a set of pauli rotation operations on single qubits. This is done by either the Bravi-Kitaev (BK) or Jordan-Wigner (JW) transformations. The result of the JW transformed Hamiltonian in the STO-6G basis is as follows:
# 
# $H = c_0 I
# + c_1 (Z_0 + Z_1) 
# + c_2 (Z_2 + Z_3)
# + c_3 (Z_0 Z_1)
# + c_4 (Z_0 Z_2 + Z_1 Z_3)
# + c_5 (Z_1 Z_2 + Z_0 Z_3)
# + c_6 (Z_2 Z_3)
# + c_7 (Y_0 X_1 X_2 Y_3 - X_0 X_1 Y_2 Y_3 - Y_0 Y_1 X_2 X_3 + X_0 Y_1 Y_2 X_3)$
# 
# Where $X_n, Y_n, Z_n$ are a pauli matrices acting on the $n$th qubit, and $c_n$ are coefficients which depend on the radial H-H distance.
# 
# The Hamiltonian is implemented in qutip below: 

# In[26]:


# # Reproducing the test set based on the STO-6G minimum basis set of hydrogen
# # From paper:

# def makepaulin(N,P):
#     PN = []
#     for n in range(N):
#         tmp = [qeye(2) for _ in range(N)]
#         tmp[n] = P
#         Pn = tensor(tmp)
#         PN.append(Pn)
#     return PN

# N=4
# X = makepaulin(N,sigmax())
# Y = makepaulin(N,sigmay())
# Z = makepaulin(N,sigmaz())
# IN = tensor([qeye(2) for _ in range(N)])

# # setting all coefficients to 1
# [c0,c1,c2,c3,c4,c5,c6,c7] = [1,1,1,1,1,1,1,1]
    
# H = (c0*IN + 
# c1*(Z[0]+Z[1]) + 
# c2*(Z[2]+Z[3]) + 
# c3*Z[0]*Z[1] + 
# c4*(Z[0]*Z[2] + Z[1]*Z[3]) + 
# c5*(Z[1]*Z[2]+Z[0]*Z[3]) + 
# c6*(Z[2]*Z[3]) + 
# c7*(Y[0]*X[1]*X[2]*Y[3] - X[0]*X[1]*Y[2]*Y[3] - Y[0]*Y[1]*X[2]*X[3] + X[0]*Y[1]*Y[2]*X[3]))

# print(H)


# In order to advance from here, we need to compute the coefficients of this Hamiltonian (which depend on radial distance between H atoms), and solve the Schroedinger equation to find ground state energies (in Hartrees) at different distances to form the set of training states. 
# 
# I think this requires an optimized solver of Schroedinger's equation, which is beyond probably beyond the scope of this project to implement. We could also use a variational quantum eigensolver, but that requires it's own detailed investigation as well.
# 
# One potential workaround could be to use PyQuante - a python library which is built to perform these kinds of caluculations (STO-6G hydrogen minimal basis set approximation). It's a dated python module, and thus far I've had version control issues getting it to integrate here.
# 
# I have requested a chemsitry textbook from the library which might lay out the calculation in more detail for us.
# 
# Until then we might need to think of a simpler test set or reach out to Zapata.

# ### Using Data from OpenFermion!!
# the bond lengths are stored in `bond_lengths.hdf5`
# 
# each bond-length has a hamiltonian stored at `hamiltonians/sto-3g.<bond-length>.hdf5`

# In[27]:


# get the list of bond lengths for which we precomputed hamiltonians
with h5py.File("bond_lengths.hdf5", "r") as f:
    try:
        dset = f['bond_lengths']
        bond_lengths = np.zeros(dset.shape)
        dset.read_direct(bond_lengths)
    except KeyError:
        print("subgroup `bond_lengths` not in `bond_lengths.hdf5` data")
        
if type(bond_lengths) == type(None):
    print("failure to read bond lengths")
else:
    bond_lengths = np.round(bond_lengths, 2)


# In[51]:


# list of hamiltonians
hamiltonians = []

for bond_length in bond_lengths:
    hamiltonian = None
    
    # read hamiltonian from appropriate file
    with h5py.File("hamiltonians/sto-3g.{}.hdf5".format(round(bond_length, 2)), 'r') as f:
        try:
            dset = f['hamiltonian']
            hamiltonian = dset[()]
            #print(hamiltonian.shape)
            #dset.read_direct(hamiltonian)
        except KeyError:
            print("subgroup `hamiltonian` not in `hamiltonians/{}` data".format(round(bond_length, 2)))
    
    if type(hamiltonian) == type(None):
        print("failure to read hamiltonian for bond length {}".format(round(bond_length, 2)))
        continue
    
    # add hamiltonian to dictionary of hamiltonians
    hamiltonians.append(Qobj(hamiltonian))
    
hamiltonians = np.array(hamiltonians)


# In[52]:


def get_groundstate(hamiltonian):
    return hamiltonian.groundstate()
groundstatize = np.vectorize(get_groundstate)


# In[79]:


# create density matrix dictionary
groundenergies, groundstates = groundstatize(hamiltonians)
training_indices = [5, 7, np.argmin(groundenergies), 20, 30, 37]


# In[82]:


# plot hydrogen atom bond-length vs. ground-state energies
# red points are those states selected for training, everything else is for testing
plt.plot(bond_lengths, groundenergies, 'ob-')
plt.plot(bond_lengths[training_indices], groundenergies[training_indices], 'or')
plt.ylim([-1.2, -0.8])
plt.xlim([0, 2.5])


# In[84]:


# generate density matrices for each groundstate
focks = []
for i in range(len(bond_lengths)):
    focks.append(ket2dm(groundstates[i]))
focks = np.array(focks)

