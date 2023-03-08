#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-03-08T15:44:41.042Z
@author: Andy Park
"""

import numpy as np
import flexIk
import time


def test_findScaleFactor():
    low = -1
    upp = 1
    a = 1
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1
    assert sMin == 0

    low = -1
    upp = 1
    a = -1
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1
    assert sMin == 0

    low = -1
    upp = 1
    a = 2
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 0.5
    assert sMin == 0

    low = -1
    upp = 1
    a = -2
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 0.5
    assert sMin == 0

    low = -1
    upp = 1
    a = 0.5
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1
    assert sMin == 0

    low = -1
    upp = 1
    a = -0.5
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1
    assert sMin == 0

    low = 0.4
    upp = 1
    a = 1.6
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 0.625
    assert sMin == 0.25

    low = -1.6
    upp = -0.4
    a = -2
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 0.8
    assert sMin == 0.2

    low = -1
    upp = 1
    a = 0
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1.0
    assert sMin == 0.0

    low = 0
    upp = 0
    a = 0
    sMax, sMin = flexIk.findScaleFactor(low, upp, a)
    assert sMax == 1.0
    assert sMin == 0.0


import time


def runTest_esnsVelIk(solver, nTest, optTol, cstTol, fid):
    """
    Run the tests for the esnsVelIk solver.
    Generates random problems for a specified test size
    and solves them using the solver, and outputs the results.
    """

    # run the tests
    nPass = 0
    nFail = 0
    nSubOptimal = 0
    solveTime = np.zeros(nTest)
    sDataTotal = [None] * nTest

    np.random.seed(3247824738)

    for iTest in range(nTest):
        # print(f'--------------- TEST {iTest + 1} ---------------\n')

        # set up the problem dimensions
        k = np.random.randint(1, 3)
        nJnt = np.random.randint(k + 2, 10)

        # set up the problem itself
        JLim = np.random.randn(k, nJnt)  # jacobian for Cartesian limits

        dqLow = -0.1 - np.random.rand(nJnt)
        dqUpp = 0.1 + np.random.rand(nJnt)

        dxLow = -0.5 - 2 * np.random.rand(k)
        dxUpp = 0.5 + 2 * np.random.rand(k)

        nTask = 2 + np.random.randint(1, 3)  # number of tasks
        ndxGoal = [None] * nTask
        scale = np.ones((nTask, 1))
        JData = [None] * nTask
        dqTmp = np.zeros((nJnt, nTask))
        dxGoalData = [None] * nTask

        for iTask in range(nTask):
            ndxGoal[iTask] = np.random.randint(1, nJnt - k)  # task space dimension

            # set up the problem itself
            JData[iTask] = np.random.randn(ndxGoal[iTask], nJnt)  # jacobian

            # compute the unscaled solution
            dqTmp[:, iTask] = dqLow + (dqUpp - dqLow) * np.random.rand(nJnt)

            for iJnt in range(nJnt):
                # saturate some joints to either limit at some probability
                if np.random.rand() < 0.2:
                    if np.random.rand() < 0.5:
                        dqTmp[iJnt, iTask] = dqLow[iJnt]
                    else:
                        dqTmp[iJnt, iTask] = dqUpp[iJnt]

            if np.random.random() < 0.2 and iTask > 1:
                dxGoalData[iTask] = np.zeros((ndxGoal[iTask], 1))
            else:
                if np.random.random() < 0.4:
                    scale[iTask] = 0.1 + 0.8 * np.random.random()
                dxGoali = JData[iTask].dot(dqTmp[:, iTask]) / scale[iTask]
                dxGoalData[iTask] = dxGoali.transpose()

        C = np.vstack((np.eye(nJnt), JLim))
        limLow = np.hstack((dqLow, dxLow))
        limUpp = np.hstack((dqUpp, dxUpp))

        # solve the problem
        testPass = True
        startTime = time.time()

        dq, sData, exitCode = solver(C, limLow, limUpp, dxGoalData, JData)
        solveTime[iTest] = time.time() - startTime
        sDataTotal[iTest] = sData
        sPrimaryTask = sDataTotal[iTest][0]

        # check the solution
        if exitCode != 1:
            print(f"\n  solver failed with exit code {exitCode}\n")
            testPass = False
        z = C.dot(dq)
        if np.any(z > limUpp + cstTol) or np.any(z < limLow - cstTol):
            print("\n  constraints are violated!  [limLow, C*dq, limUpp]")
            testPass = False
        taskError = sPrimaryTask * dxGoalData[0] - JData[0].dot(dq)
        if np.any(np.abs(taskError) > cstTol):
            print(
                "\n  task error (%.6f) exceeds tolerance!" % np.max(np.abs(taskError))
            )
            testPass = False

        if testPass:
            nPass = nPass + 1
        else:
            nFail = nFail + 1
            print(f"--------------- TEST {iTest + 1} ---------------")
            print("  --> fail\n")

    print("\n")
    print("------------------------------------")
    print("nPass: %d  --  nFail: %d" % (nPass, nFail))
    print("------------------------------------")

    result = dict()
    result["nPass"] = nPass
    result["nFail"] = nFail
    result["nSubOptimal"] = nSubOptimal
    result["solveTime"] = solveTime
    result["sDataTotal"] = sDataTotal

    return result


def test_velIkSolver():
    # shared test parameters
    optTol = 1e-4
    cstTol = 1e-6
    fid = 1
    nTest = 1000

    # extended SNS-IK solver algorithm for multiple tasks
    result = runTest_esnsVelIk(flexIk.velIk, nTest, optTol, cstTol, fid)
    sDataTotal = result["sDataTotal"]
    sDataPrimary = np.zeros(nTest)
    sDataSecondary = np.zeros(nTest)
    numTaskData = np.zeros(nTest)
    numTaskFeasibleData = np.zeros(nTest)
    for i in range(nTest):
        sDataTmp = sDataTotal[i]
        sDataPrimary[i] = sDataTmp[0]
        sDataSecondary[i] = sDataTmp[1]
        numTaskData[i] = len(sDataTmp)
        numTaskFeasibleData[i] = len(np.where(np.array(sDataTmp) > 0)[0])
    print(f"average primary task scale factor: {np.mean(sDataPrimary):.4f}")
    print(f"average secondary task scale factor: {np.mean(sDataSecondary):.4f}")
    print(
        f"average number of feasible tasks: ({np.mean(numTaskFeasibleData):.4f}/{np.mean(numTaskData):.4f})"
    )
    print(
        f"percentage of sub-optimal task scale factors: {result['nSubOptimal']/nTest*100:.2f}%"
    )

    assert result["nFail"] == 0
