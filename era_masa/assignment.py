import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(cost, valid, maximum_cost, greedy=False):
    """Gate before assignment; dummy columns permit every detection to abstain.

    Post-Hungarian thresholding can discard a pair that stole a column from a
    valid pair. Invalid edges must therefore be removed before optimization.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    valid = valid & np.isfinite(cost) & (cost < maximum_cost)
    if greedy:
        matches, used = [], set()
        for row in range(n):
            choices = np.where(valid[row], cost[row], np.inf)
            for col in used:
                choices[col] = np.inf
            col = int(np.argmin(choices))
            if np.isfinite(choices[col]):
                matches.append((row, col))
                used.add(col)
        return matches
    padded = np.full((n, m + n), maximum_cost, dtype=float)
    padded[:, :m] = np.where(valid, cost, maximum_cost + 1e6)
    rows, columns = linear_sum_assignment(padded)
    return [(int(r), int(c)) for r, c in zip(rows, columns) if c < m and valid[r, c]]
