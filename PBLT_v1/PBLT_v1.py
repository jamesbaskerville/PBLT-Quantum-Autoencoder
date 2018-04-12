
# coding: utf-8

# In[1]:


import numpy as np
from qutip import *


# #### Parameters
# 
# For n qubits, we need n(n-1) + 2n arbitary rotation gates, with 3 parameters each.
# 
# https://arxiv.org/abs/1612.02806

# In[2]:


n = 4
num_gates = n * (n - 1) + 2 * n
num_params = 3 * num_gates


# ### Qutip Implementation

# In[3]:


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


# In[62]:


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
    print(n, params, ctrl, tgt)
    alpha, theta, beta = params
    A = rz(alpha, n, tgt) * ry(theta / 2.0, n, tgt)
    B = ry(-theta / 2.0, n, tgt) * rz(-(alpha + beta) / 2.0, n, tgt)
    C = rz((beta - alpha) / 2.0, n, tgt)
    assert (A*B*C == tensor(qeye(2), tensor(qeye(2), tensor(qeye(2), qeye(2)))))
    assert (A*tgtgate(n, sigmax(), tgt)*B*tgtgate(n, sigmax(), tgt)*C == arb_rot(n, params, tgt))
    return A * cnot(n, ctrl, tgt) * B * cnot(n, ctrl, tgt) * C


# #### Data Structures

# In[64]:


def split_params(n, params):
    return (params[:3*n].reshape(n, 3),
            params[3*n:-3*n].reshape(n, n-1, 3),
            params[-3*n:].reshape(n, 3))

def recombine_params(first, mid, last):
    return np.concatenate((first.flatten(), mid.flatten(), last.flatten()))


# #### Circuit
# The circuit outlined in red below is the unitary gate for encoding (in this case, for 4 qubit inputs).
# ![arbitrary_rotation_gate_circuit](https://image.ibb.co/ji9XBc/unit_cell_arb_rot.png)

# In[14]:


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


# In[16]:


# gate_product = gate_prod(n, gates)
# gate_product


# In[73]:


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


# In[78]:


# testing
params = np.ones(num_params)
U = create_circuit(n, params)
# for ctrl, blue_box_params in enumerate(m_params):
#     blue_box(n, ctrl, blue_box_params)
U


# In[ ]:


U = rz(1)*ry(2)*rz(3)
ctrl_rot(4, (1,2,3), 0, 1) == controlled_gate(U, 4, 0, 1)

