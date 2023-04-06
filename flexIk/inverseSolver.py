import numpy as np
from numpy.linalg import pinv
from enum import Enum


class Types(Enum):
    PINV = 1
    DLS = 2
    SRINV = 3


def inv(A, method=Types.PINV, tolerance=1e-1, betaMax=1e-1):
    """
    Compute inverse of matrix A using different methods.

    Parameters:
        A (numpy.ndarray): Input matrix to be inverted.
        method (InverseMethods): Method of inversion (default=Types.PINV).
        tolerance (float): Tolerance value for damping methods (default=1e-4).
        betaMax (float): Maximum damping coefficient (default=1e-1).

    Returns:
        A_inv (numpy.ndarray): Inverse of matrix A.

    """

    if method == Types.PINV:
        # method 1: matlab's pinv with a threshold
        A_inv = np.linalg.pinv(A, rcond=1e-8)

    elif method == Types.DLS:
        # method 2: damped least-square inverse
        lambda_val = tolerance

        A_inv = A.T @ np.linalg.inv(A @ A.T + lambda_val**2 * np.eye(A.shape[0]))

    elif method == Types.SRINV:
        # method 3: singularity-robust inverse base on numerical filtering

        # initialize some constants
        beta = 0
        lambda_val = 0

        # get the minimum singular value from SVD
        U, S, V = np.linalg.svd(A)

        s_min = np.min(S)

        # compute the sum of the output vectors that correspond to
        # singular values below the threshold (tolerance)
        u_sum = np.zeros((A.shape[0], A.shape[0]))
        if s_min < tolerance:
            for ind in range(S.size - 1, -1, -1):
                if S[ind] < tolerance:
                    u_sum += np.outer(U[:, ind], U[:, ind])
                else:
                    break
            # scale beta coefficient with the minimum singular value
            # beta maps to the range of [0, betaMax] depending on the size of
            # minimum singular value with respect to the threshold.
            beta = (1 - (s_min / tolerance) ** 2) * betaMax

        # compute the inverse with beta coefficient
        A_inv = A.T @ np.linalg.inv(
            A @ A.T + lambda_val**2 * np.eye(A.shape[0]) + beta * u_sum
        )

    return A_inv
