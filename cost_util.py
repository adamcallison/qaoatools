import numpy as np
import quimb as qu

import random
from functools import lru_cache

Z = qu.pauli('z',2,sparse=True, dtype=np.float64)
X = qu.pauli('x',2,sparse=True, dtype=np.float64)
Identity = qu.pauli('i', 2, sparse=True, dtype=np.float64)

def cost_hamiltonian(J, h, ic):
    n1 = J.shape[0]
    n2 = J.shape[1]
    n3 = h.shape[0]
    assert n1 == n2
    assert n1 == n3
    assert n2 == n3
    n = n1

    Hp = qu.ikron(ic*Identity, dims=(2,)*n, inds=(n-1,), sparse=True)
    for i in range(n):
        Hp += qu.ikron(h[i]*Z, dims=(2,)*n, inds=((n-i)-1,), sparse=True)
        for j in range(n):
            if i==j: continue
            Hp += qu.ikron([J[i,j]*Z, Z], dims=(2,)*n, inds=((n-i)-1, (n-j)-1), sparse=True)

    return Hp

def cost_eigenvalues(J, h, c):
    n = h.shape[0]
    N = 2**n
    costs = np.ndarray(N)
    costs[:] = c
    states = (N - 1) -  np.arange(N)
    for q1 in range(n):
        q1bit = (states & (1 << q1)) >> q1
        costs += -((-1)**q1bit)*h[q1]
        for q2 in range(n):
            q2bit = (states & (1 << q2)) >> q2
            costs += ((-1)**(q1bit + q2bit))*J[q1, q2]
    return costs

def adjacency_to_maxcut(A, cutoff=0.0):
    n = A.shape[0]
    J, h, c = np.zeros((n, n)), np.zeros(n), 0.0
    for j in range(n):
        for k in range(n):
            if np.abs(A[j, k]) > np.abs(cutoff):
                c += 0.5*A[j, k]
                J[j, k] += 0.5*A[j, k]
    return J, h, c

def erdosrenyi_maxcut(n_nodes, edge_probability):
    n = n_nodes
    p = edge_probability
    A = np.zeros((n, n))
    for j in range(n-1):
        for k in range(j+1, n):
            A[j, k] = np.random.default_rng().choice([0.0, 1.0], p=[1-p, p])

    J, h, c = adjacency_to_maxcut(A)
    
    return A, (J, h, c)

def isingify_m2s(nqubits, prob):
    J = np.zeros((nqubits, nqubits))
    h = np.zeros(nqubits)
    c = 0.0

    for i, clause in enumerate(prob):
        s0, v0, s1, v1 = tuple([int(x) for x in clause])

        c += 0.25
        J[v0, v1] += 0.25*s0*s1
        h[v0] -= 0.25*s0
        h[v1] -= 0.25*s1

    return J, h, c

@lru_cache(maxsize=1)
def max2sat_all_clauses(nqubits):
    all_clauses = []
    for s1 in (-1, 1):
        for q1 in range(nqubits):
            for s2 in (-1, 1):
                for q2 in range(nqubits):
                    if q1 == q2:
                        continue
                    all_clauses.append((s1, q1, s2, q2))
    all_clauses = sorted(all_clauses, key=lambda x:(x[1]*nqubits)+x[3])
    return all_clauses

def max2sat_prob(nqubits=None, nclauses=None):
    ac = max2sat_all_clauses(nqubits)
    cs = random.sample(ac, nclauses)
    cs = sorted(cs, key=lambda x:(x[1]*nqubits)+x[3])
    return np.array(cs)
