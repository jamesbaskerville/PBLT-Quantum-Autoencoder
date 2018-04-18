
# coding: utf-8

# In[32]:


import numpy as np
from qutip import *
from scipy.optimize import minimize


# #### Parameters
# 
# For n qubits, we need n(n-1) + 2n arbitary rotation gates, with 3 parameters each.
# 
# https://arxiv.org/abs/1612.02806

# ### Qutip Implementation

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


# In[16]:


# arbitrary rotation/controlled rotation gates
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


# #### Data Structures

# In[37]:


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


# In[7]:


def wrapper(n, params):
    assert (len(params) == n)
    gates = []
    for tgt, rot_params in enumerate(params):
        gates.append(rot(n, rot_params, tgt))
    return gate_prod(n, gates)

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


# In[132]:


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


# In[117]:


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


# In[99]:


data = gen_data(2,1,data_count=1000)


# In[103]:


initial_params = init_params(num_params)
data[:5]
# minimize(obj_func, initial_params, method='Nelder-Mead')


# In[133]:


U = sigmax()
instates = np.array([Qobj([[0],[1]]), Qobj([[1],[0]])])
outstates = U*instates
sum(1-v_overlap(instates, outstates))


# In[114]:


qubit()


# In[115]:


overlap()

