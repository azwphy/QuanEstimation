import pytest
import numpy as np
from quanestimation.Common.Common import basis, gramschmidt, suN_generator

def test_basis():
    """
    Test the basis function for generating quantum state basis vectors.
    This test checks the generation of basis vectors for a 2-dimensional quantum system.
    """
    result = basis(2, 0)
    assert np.allclose(result, np.array([[1], [0]])) == 1

def test_gramschmidt():
    """
    Test the Gram-Schmidt process for orthonormalizing a set of vectors.
    This test checks the orthonormalization of a set of vectors in a 2-dimensional complex space.
    """
    A = np.array([[1., 0.], [1., 1.]], dtype=np.complex128)
    result = gramschmidt(A)
    assert np.allclose(result, np.array([[1.+0.j, 0.+0.j],[0.+0.j, 1.+0.j]])) == 1

def test_suN_generator():
    """
    Test the generation of SU(N) generators for a 2-dimensional quantum system.
    This test checks the correctness of the SU(2) generators, which are the Pauli matrices.
    """
    n = 2
    result = suN_generator(n)
    sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128) 
    sy = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
    sz = np.array([[1., 0.j], [0.j, -1.]], dtype=np.complex128)
    assert (np.allclose(result[0], sx) and np.allclose(result[1], sy) and \
            np.allclose(result[2], sz)) == 1    
    
    