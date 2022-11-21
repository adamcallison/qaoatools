import numpy as np
from functools import lru_cache
import quimb as qu
Z = qu.pauli('z',2,sparse=True, dtype=np.float64)
X = qu.pauli('x',2,sparse=True, dtype=np.float64)
Identity = qu.pauli('i', 2, sparse=True, dtype=np.float64)

@lru_cache(maxsize=1)
def standard_mixer_eigenvalues(n):
    # returns diagonal of standard mixer that has been hadamard Hd_transformed
    # into Z eigenbasis
    N = 2**n
    eigvals = np.zeros(N)
    for j in range(N):
        jstr = bin(j)[2:]
        ones = jstr.count('1')
        eigvals[j] = 2*ones
    return eigvals

@lru_cache(maxsize=1)
def standard_mixer_hamiltonian(n):
    Hw = qu.ikron(n*Identity, dims=(2,)*n, inds=(n-1,), sparse=True)
    for i in range(n):
        Hw -= qu.ikron(X, dims=(2,)*n, inds=((n-i)-1,), sparse=True)
    return Hw