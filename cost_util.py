import numpy as np
import quimb as qu
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

    Hp = qu.ikron(ic*Identity, dims=(2,)*n, inds=(0,), sparse=True)
    for i in range(n):
        Hp += qu.ikron(h[i]*Z, dims=(2,)*n, inds=(i,), sparse=True)
        for j in range(n):
            if i==j: continue
            Hp += qu.ikron([J[i,j]*Z, Z], dims=(2,)*n, inds=(i, j), sparse=True)

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

def erdosrenyi_maxcut(n_nodes, edge_probability):
    n = n_nodes
    p = edge_probability
    A = np.zeros((n, n))
    for j in range(n-1):
        for k in range(j+1, n):
            A[j, k] = np.random.default_rng().choice([0.0, 1.0], p=[1-p, p])
    
    J, h, c = np.zeros((n, n)), np.zeros(n), 0.0
    for j in range(n):
        for k in range(n):
            if A[j, k] == 1:
                c += 0.5
                J[j, k] += 0.5

    return A, (J, h, c)