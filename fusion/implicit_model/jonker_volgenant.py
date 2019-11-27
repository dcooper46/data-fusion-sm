# Joneker-Volgenant LAP solution
import numpy as np


MAXVAL = np.finfo(np.float).max


def lapjv(costs):
    """ """
    nrows, ncols = costs.shape

    numfree = 0
    j = 0
    j1 = 0
    j2 = 0
    i0 = 0
    freerow = 0
    endOfPath = 0
    h = 0
    minn = 0

    free = np.zeros(nrows, dtype=int)
    pred = np.zeros(nrows, dtype=int)
    collist = np.zeros(nrows, dtype=int)
    matches = np.zeros(nrows, dtype=int)
    v = np.zeros(nrows, dtype=int)
    d = np.zeros(nrows, dtype=int)
    colSol = np.zeros(nrows, dtype=int)
    rowSol = np.zeros(ncols, dtype=int)

    # column reduction
    for j in range(ncols - 1, -1, -1):
        imin = np.argmin(costs[:, j])
        minn = np.min(costs[:, j])
        v[j] = minn
        matches[imin] += 1
        if matches[imin] == 1:
            rowSol[imin] = j
            colSol[j] = imin
        else:
            colSol[j] = -1

    # reduction transfer
    for i in range(nrows):
        if matches[i] == 0:
            free[numfree] = i
            numfree += 1
        else:
            if matches[i] == 1:
                j1 = rowSol[i]
                minn = MAXVAL
                for j in range(nrows):
                    if (j != j1) and (costs[i, j] - v[j] < minn):
                        minn = costs[i, j] - v[j]
                v[j1] -= minn

    for _ in (1, 2):
        k = 0
        prevNumfree = numfree
        numfree = 0
        while (k < prevNumfree):
            i = free[k]
            k += 1

            umin = costs[i, 0] - v[0]
            j1 = 0
            usubmin = MAXVAL
            for j in range(1, nrows):
                h = costs[i, j] - v[j]
                if h < usubmin:
                    if h >= umin:
                        usubmin = h
                        j2 = j
                    else:
                        usubmin = umin
                        umin = h
                        j2 = j1
                        j1 = j

            i0 = colSol[j1]
            if umin < usubmin:
                v[j1] -= (usubmin - umin)
            else:
                if i0 >= 0:
                    j1 = j2
                    i0 = colSol[j2]

            rowSol[i] = j1
            colSol[j1] = i

            if i0 >= 0:
                if umin < usubmin:
                    k -= 1
                    free[k] = i0
                else:
                    free[numfree] = i0
                    numfree += 1

    # Augment Solution for each free row
    for f in range(numfree):
        freerow = free[f]

        # Dijkstra shortest path algorithm.
        # runs until unassigned column added to shortest path tree.
        for j in range(nrows):
            d[j] = costs[freerow, j] - v[j]
            pred[j] = freerow
            collist[j] = j

        low = 0
        up = 0
        last = 0

        unassignedFound = False
        while not unassignedFound:
            if up == low:  # no more columns to be scanned for current minimum.
                last = low - 1
                # scan columns for up..dim-1 to find all indices for which new minimum occurs.
                # store these indices between low..up-1 (increasing up).
                minn = d[collist[up]]
                up += 1
                for k in range(up, nrows):
                    j = collist[k]
                    h = d[j]
                    if h <= minn:
                        if h < minn:
                            up = low
                            minn = h
                        # new index with same minimum, put on undex up, and extend list.
                        collist[k] = collist[up]
                        collist[up] = j
                        up += 1

                for k in range(low, up):
                    if colSol[collist[k]] < 0:
                        endOfPath = collist[k]
                        unassignedFound = True
                        break

            # update 'distances' between freerow and all unscanned columns, via next scanned column.
            if unassignedFound:
                break
            else:
                j1 = collist[low]
                low += 1
                i = colSol[j1]
                h = costs[i, j1] - v[j1] - minn
                for k in range(up, nrows):
                    j = collist[k]
                    v2 = costs[i, j] - v[j] - h
                    if v2 < d[j]:
                        pred[j] = i
                        if v2 == minn:  # new column found at same minimum value
                            # if unassigned, shortest augmenting path is complete.
                            if colSol[j] < 0:
                                endOfPath = j
                                unassignedFound = True
                                break
                            else:  # else add to list to be scanned right away.
                                collist[k] = collist[up]
                                collist[up] = j
                                up += 1
                                d[j] = v2
                        else:
                            d[j] = v2

        # update column prices.
        for k in range(last+1):
            j1 = collist[k]
            v[j1] = v[j1] + d[j1] - minn

        # reset row and column assignments along the alternating path.
        i = 0
        while True:
            i = pred[endOfPath]
            colSol[endOfPath] = i
            j1 = endOfPath
            endOfPath = rowSol[i]
            rowSol[i] = j1
            if i == freerow:
                break

    # calculate optimal cost
    final_cost = sum([costs[i, rowSol[i]] for i in range(nrows)])
    rows = np.array([range(nrows)])

    return rows, rowSol, final_cost
