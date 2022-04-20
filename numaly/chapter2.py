import numpy as np

class GaussJordan(object):
    """Performing Gauss elimination to solve the matrix equation `Ax = b`."""

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        assert len(A.shape) == 2, "A must be a 2D matrix"
        assert A.shape[0] == A.shape[1], "A must be a 2D square matrix"

        self.n = self.A.shape[0]

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        @param `b`: the given vector.
        @return `x`: solution to the equation Ax = b.
        """
        x = np.array(b, dtype=float)

        # last row is naturally reduced 
        for i in range(self.n - 1):
            assert (np.abs(self.A[i, i]) > 1e-10), "zero pivot encountered"
            # reduce one row with the multiplication of top row
            for j in range(i + 1, self.n):
                mul = self.A[j, i] / self.A[i, i]
                # need not place 0 at the left of pivot
                self.A[j, i:] -= mul * self.A[i, i:]
                x[j] -= mul * x[i]

        # back substitution
        x[-1] /= self.A[-1, -1]
        for i in range(-2, -self.n - 1, -1):
            x[i] = (x[i] - (self.A[i, i + 1: ] @ x[i + 1: ])) / self.A[i, i]

        return x

class LU(object):
    """
    Decompsite the given matrix into the multiplication of an upper triangular matrix and a lower triangular matrix,
    that is `A = LU`.
    """
    def __init__(self, A: np.ndarray):
        """
        @param `A`: the given matrix.
        @return `L, U`: the triangular matrices.
        """
        A = np.asarray(A, dtype=float)
        assert len(A.shape) == 2, "A must be a 2D matrix"
        assert A.shape[0] == A.shape[1], "A must be a 2D square matrix"

        self.n = A.shape[0]
        self.L = np.eye(self.n, dtype=A.dtype)
        self.U = np.zeros_like(A, dtype=A.dtype)

        # for first row of U and first column of L
        self.U[0, :] = A[0, :]
        self.L[1:, 0] = A[1:, 0] / self.U[0, 0]

        # for each row of U and each column of L
        for i in range(1, self.n - 1):
            self.U[i, i:] = A[i, i:] - (self.L[i, :i] @ self.U[:i, i:])
            self.L[i + 1:, i] = (A[i + 1:, i] - (self.L[i + 1:, :i] @ self.U[:i, i])) / self.U[i, i]

        # for the last element of U
        self.U[-1, -1] = A[-1, -1] - (self.L[-1, :-1] @ self.U[:-1, -1])


    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Using LU decomposition to solve the matrix equation Ax = b.
        Or Ly = b, Ux = y in detail.
        @param `b`: the given matrix and vector.
        @return `x`: solution to the equation Ax = b.
        """
        x = np.zeros_like(b, dtype=float)
        # forward substitution of L
        x[0] = b[0]
        for i in range(1, self.n):
            x[i] = b[i] - (self.L[i, :i] @ x[:i])

        # back substitution of U
        x[-1] /= self.U[-1, -1]
        for i in range(-2, -self.n - 1, -1):
            x[i] = (x[i] - (self.U[i, i + 1: ] @ x[i + 1: ])) / self.U[i, i]

        return x


class Cholesky(object):
    """
    Decompsite the given symmetric matrix into the multiplication of a lower triangular matrix and its transpose,
    that is `A = RR^T`.
    """
    def __init__(self, A: np.ndarray):
        """
        @param A: the given matrix.
        @return: the lower triangular matrices.
        """
        A = np.asarray(A, dtype=float)
        assert len(A.shape) == 2, "A must be a 2D matrix"
        assert A.shape[0] == A.shape[1], "A must be a 2D square matrix"
        assert np.array_equal(A, A.T), "A must be a symmetric matrix"
        n = A.shape[0]

        self.R = np.zeros_like(A, dtype=A.dtype)
        # for first row and column of R
        self.R[0, 0] = np.sqrt(A[0, 0])
        self.R[0, 1:] = A[0, 1:] / self.R[0, 0]
        self.R[1:, 0] = A[1:, 0] / self.R[0, 0]

        # for each row and column of R
        for i in range(1, n - 1):
            self.R[i, i] = np.sqrt(A[i, i] - (self.R[i, :i] @ self.R[:i, i]))
            self.R[i, i + 1:] = (A[i, i + 1:] - (self.R[i, :i] @ self.R[:i, i + 1:])) / self.R[i, i]
            self.R[i + 1:, i] = (A[i + 1:, i] - (self.R[i + 1:, :i] @ self.R[:i, i])) / self.R[i, i]

        # for the last element of R
        self.R[-1, -1] = np.sqrt(A[-1, -1] - (self.R[-1, :-1] @ self.R[:-1, -1]))
        self.R = np.tril(self.R)
