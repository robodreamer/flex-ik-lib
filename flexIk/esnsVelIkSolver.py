import numpy as np
from numpy.linalg import pinv

def findScaleFactor(low, upp, a):
    """
    FINDSCALEFACTOR computes feasible task scale factors for SNS IK algorithms.

    This function computes task scale factors from upper and lower margins
    and the desired task for a single component. This function is called by
    all of SNS IK algorithms.

    INPUTS:
        low (float): lower margin
        upp (float): upper margin
        a (float): desired task

    OUTPUTS:
        sMax (float): the maximum feasible scale factor [0, 1]
        sMin (float): the minimum feasible scale factor [0, 1]

    NOTES:
    """

    sMax = 1 # the maximum feasible scale factor
    sMin = 0 # the minimum feasible scale factor
    if abs(a) < 1e10 and abs(a) > 1e-10:
        if a < 0 and low < 0 and a <= upp:
            sMax = min(1, low/a)
            sMin = max(0, upp/a)
        elif a >= 0 and upp > 0 and a >= low:
            sMax = min(1, upp/abs(a))
            sMin = max(0, low/abs(a))
    return sMax, sMin


def esns_velocity_ik(C, limLow, limUpp, dxGoalData, JData, invSolver=None):
    """
    Calculates joint velocities given a set of Jacobians and task goals
    using the inverse kinematics algorithm with velocity-based multi-task
    prioritization and constraints. This function uses a PINV solver to
    compute the pseudoinverse of the Jacobian matrix.

    Args:
    C (numpy.ndarray): Joint limits constraint matrix.
    limLow (numpy.ndarray): Joint lower limits.
    limUpp (numpy.ndarray): Joint upper limits.
    dxGoalData (list): List of numpy arrays, each containing the desired
        velocity for each task.
    JData (list): List of numpy arrays, each containing the Jacobian matrix
        for each task.
    invSolver (callable, optional): A function that computes the inverse of
        a matrix. Defaults to None, which sets the inverse solver to use the
        PINV method.

    Returns:
    tuple: A tuple containing the following elements:
        dq (numpy.ndarray): The joint velocities that achieve the task goals.
        sData (numpy.ndarray): An array containing the task scaling factors
            used to achieve the task goals.
        exitCode (int): A flag indicating whether the algorithm succeeded
            (1) or failed (0).
    """

    # use PINV method if the invMethod not specified
    if invSolver is None:
        invSolver = lambda A: inverseSolver(A, InverseMethods.PINV)

    # get the number of tasks
    nTask = len(JData)
    sData = np.zeros(nTask)

    # initialization
    exitCode = 1
    tol = 1e-6 # a tolerance used for various numerical checks
    tolTight = 1e-10 # a tighter tolerance
    nIterationsMax = 20

    nJnt = JData[0].shape[1]
    k = C.shape[0] - nJnt
    I = np.eye(nJnt)
    Pi = I
    Sact = np.zeros((nJnt + k, nJnt + k))
    dqi = np.zeros((nJnt, 1))
    w = np.zeros((nJnt + k, 1))
    Cact = Sact.dot(C)

    for iTask in range(nTask):

        # get i-th task jacobian
        Ji = JData[iTask]
        ndxGoal = Ji.shape[0]

        # get i-th task velocity
        dxGoali = dxGoalData[iTask].reshape(-1,1)

        # update variables for previous projection matrix and solution
        PiPrev = Pi
        dqiPrev = dqi

        # initialize variables for i-th task
        PiBar = PiPrev
        si = 1.0
        siStar = 0.0

        limitExceeded = True
        cntLoop = 1

        SactStar = Sact
        wStar = w

        PiBarStar = np.zeros_like(PiBar)
        PiHatStar = np.zeros((nJnt, nJnt + k))

        if np.sum(np.abs(dxGoali)) < tolTight:
            dqi = dqiPrev
            limitExceeded = False

        while limitExceeded:
            limitExceeded = False

            # update PiHat
            PiHat = (I - invSolver(Ji.dot(PiBar)).dot(Ji)).dot(pinv(Cact.dot(PiPrev), tol))

            # compute a solution without task scale factor
            dqi = dqiPrev + invSolver(Ji.dot(PiBar)).dot(dxGoali - Ji.dot(dqiPrev)) + \
                PiHat.dot((w - Cact.dot(dqiPrev)))

            # check whether the solution violates the limits
            z = C.dot(dqi)
            z = np.ravel(z)
            if np.any(z < (limLow - tol)) or np.any(z > (limUpp + tol)):
                limitExceeded = True

            # compute goal velocity projected
            xdGoalProj = C.dot(invSolver(Ji.dot(PiBar))).dot(dxGoali)

            if np.linalg.norm(xdGoalProj) < tol:
                # if the projected goal velocity is close to zero, set scale
                # factor to zero
                si = 0
            else:
                # compute the scale factor and identify the critical joint
                a = xdGoalProj
                b = C.dot(dqi) - a.reshape(-1,1)

                marginL = limLow - np.ravel(b)
                marginU = limUpp - np.ravel(b)

                sMax = np.zeros(nJnt + k)
                sMin = np.zeros(nJnt + k)
                for iLim in range(nJnt + k):
                    if Sact[iLim, iLim] == 1:
                        sMax[iLim] = np.inf
                    else:
                        sMax[iLim], sMin[iLim] = \
                        findScaleFactor(marginL[iLim], marginU[iLim], a[iLim])

                # most critical limit index
                mclIdx = np.argmin(sMax)
                si = sMax[mclIdx]

            if np.isinf(si):
                si = 0

            # do the following only if the task is feasible and the scale
            # factor caculated is correct
            if (iTask == 1 or si > 0) and cntLoop < nIterationsMax:

                scaledDqi = dqiPrev + invSolver(Ji.dot(PiBar)).dot( \
                    (si * dxGoali - Ji.dot(dqiPrev))) + \
                    PiHat.dot(w - Cact.dot(dqiPrev))

                z = C.dot(scaledDqi)
                z = np.ravel(z)

                limitSatisfied = not (any(z < (limLow - tol)) or \
                        any(z > (limUpp + tol)))
                if si > siStar and limitSatisfied:
                    siStar = si
                    SactStar = Sact
                    wStar = w
                    PiBarStar = PiBar
                    PiHatStar = PiHat

                Sact[mclIdx, mclIdx] = 1
                Cact = Sact.dot(C)

                w[mclIdx,0] = min(max(limLow[mclIdx], z[mclIdx]), limUpp[mclIdx])

                PiBar = PiPrev - pinv(Cact.dot(PiPrev), tol).dot(Cact.dot(PiPrev))

                taskRank = np.linalg.matrix_rank(Ji.dot(PiBar), tol)
                if taskRank < ndxGoal:
                    si = siStar
                    Sact = SactStar
                    Cact = Sact.dot(C)
                    w = wStar
                    PiBar = PiBarStar
                    PiHat = PiHatStar

                    dqi = dqiPrev + invSolver(Ji.dot(PiBar)).dot( \
                        (si * dxGoali - Ji.dot(dqiPrev))) + \
                        PiHat.dot(w - Cact.dot(dqiPrev))

                    limitExceeded = False
            else:  # if the current task is infeasible
                si = 0
                dqi = dqiPrev
                limitExceeded = False

                if cntLoop == nIterationsMax:
                    print('\n\tWarning: the maximum number of iteration has been reached!\n')

            cntLoop = cntLoop + 1

            if (si > 0):
                # update nullspace projection
                Pi = PiPrev - pinv(Ji.dot(PiPrev), tol).dot(Ji.dot(PiPrev))

        sData[iTask] = si

    dqi = np.ravel(dqi)

    return dqi, sData, exitCode
