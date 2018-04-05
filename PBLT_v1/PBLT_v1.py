
# coding: utf-8

# In[2]:


import numpy as np
from qutip import *


# #### Parameters
# 
# For n qubits, we need n(n-1) + 2n arbitary rotation gates, with 3 parameters each.
# 
# https://arxiv.org/abs/1612.02806

# In[12]:


n = 4
num_gates = n * (n - 1) + 2 * n
num_params = 3 * num_gates


# ### Qutip Implementation

# In[16]:


# arbitrary rotation gate on one qubit
def arb_rot(ax, ay, az):
    return rx(ax)*ry(ay)*rz(az)


# #### Data Structures

# In[21]:


gates = np.array([qeye(2) for _ in range(num_gates)])
params = np.ones(num_params)


# #### Circuit
# The circuit outlined in red below is the unitary gate for encoding (in this case, for 4 qubit inputs).
# ![arbitrary_rotation_gate_circuit](https://image.ibb.co/ji9XBc/unit_cell_arb_rot.png)

# In[27]:


# R1 - R4
for gate_num in range(4):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    gates[gate_num] = arb_rot(ax, ay, az)

# R5 - R7   controlled by qubit 0
for gate_num in range(4, 7):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    #gates[gate_num] = arb_rot(ax, ay, az)

# R8 - R10  controlled by qubit 1
for gate_num in range(7, 10):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    #gates[gate_num] = arb_rot(ax, ay, az)

# R11 - R13 controlled by qubit 2
for gate_num in range(10, 13):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    #gates[gate_num] = arb_rot(ax, ay, az)

# R14 - R16 controlled by qubit 3
for gate_num in range(13, 16):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    #gates[gate_num] = arb_rot(ax, ay, az)

# R17 - R20
for gate_num in range(16, 20):
    ax, ay, az = params[3*gate_num : 3*(gate_num+1)]
    gates[gate_num] = arb_rot(ax, ay, az)

