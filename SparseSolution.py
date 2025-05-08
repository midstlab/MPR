import time
from numpy import ceil, loadtxt, linalg, inner, matmul, zeros, array, log10, floor
from gurobipy import GRB, Model, quicksum
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import heapq


class Solution:
    def __init__(self, inner, subset, weights):
        self.inner = inner
        self.subset = subset
        self.weights = weights
    def __lt__(self, other):
        return self.inner < other.inner


def Output(m):  
    status_code = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED'}
    status = m.status
    print('The optimization status is ' + status_code[status])
    if status == 2:
        print('Optimal solution:')
        for v in m.getVars():
            if abs(v.x) > 1e-5:
                print(f"{v.varName} = {v.x:.5f}")
        print('Optimal objective value:', m.objVal, "\n")


def Norm2Sq(v):
    return sum(v[i]**2 for i in range(len(v)))


def LinearSolve(C, R):
    start = time.time()
    f = linalg.solve(C, R)
    print("Time Elapsed in seconds (Linear Solve):", time.time() - start)
    return f


def PrintConsole(bestSolutions, timeElapsed):
    output_lines = ["Time Elapsed in seconds: " + str(timeElapsed)]
    for i, sol in enumerate(bestSolutions, start=1):
        output_lines.append(f"Best solution number {i}")
        output_lines.append("Value: " + str(sol.inner))
        output_lines.append("Residue Indices (starting from 1): " + str(sol.subset))
        for j in range(len(sol.subset)):
            w = sol.weights[3*j:3*j+3]
            output_lines.append(f"{w[0]},{w[1]},{w[2]}")
        output_lines.append("")
    text = "\n".join(output_lines)
    print(text)
    with open("results.txt", "a") as f:
        f.write(text + "\n")


def OutputModel(start, end, M, z, y, R, regularization):
    sol = zeros(len(R))
    subset = []
    weights = []
    for i in range(M):
        if z[i].x > 1e-3:
            subset.append(i+1)
            for h in range(3):
                j = 3*i + h
                weights.append(y[j].x)
                sol += y[j].x * C[:, j]
    return PrintConsole([
        Solution(NormalizedInnerProduct(sol, R), subset, weights)
    ], str(end - start))


def OptimizationApproach(C, R, K, bigM=100, regularization=-1):
    print("Using the OPTIMIZATION approach with k=", K, "bigM=", bigM)
    if regularization >= 0:
        print("Penalty=", regularization)
    start = time.time()
    N = len(R)
    M = N // 3
    model = Model('normalizedInnerProduct')
    model.setParam('OutputFlag', False)
    y = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='y')
    u = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    model.addConstrs(u[i] - quicksum(C[i, j]*y[j] for j in range(N)) == -R[i] for i in range(N))
    obj = quicksum(u[i]*u[i] for i in range(N))
    if regularization < 0:
        z = model.addVars(M, vtype=GRB.BINARY, name='z')
        model.addConstrs(y[i] >= -bigM*z[i//3] for i in range(N))
        model.addConstrs(y[i] <=  bigM*z[i//3] for i in range(N))
        model.addConstr(z.sum('*') == K)
        model.setParam('MIPGap', 1e-3)
    else:
        z = model.addVars(M, lb=0, ub=GRB.INFINITY, name='z')
        model.addConstrs(y[i] <=  z[i//3] for i in range(N))
        model.addConstrs(-y[i] <=  z[i//3] for i in range(N))
        obj += regularization * z.sum('*')
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    end = time.time()
    OutputModel(start, end, M, z, y, R, regularization)


def LeastSquaresSolution(C, R, indices):
    return linalg.solve(matmul(C[indices,:], C[:,indices]), matmul(C[indices,:], R))


def NormalizedInnerProduct(v, R, R_norm=-1):
    if R_norm < 0:
        R_norm = linalg.norm(R)
    return inner(v, R) / (linalg.norm(v) * R_norm)


def EnumerationApproach(C, R, K, best=10):
    print("Using the ENUMERATION approach with k=", K)
    start = time.time()
    N = len(R)
    M = N // 3
    bestSolutions = [Solution(0, [], [])] * best
    R_norm = linalg.norm(R)
    inners = []

    for subset in combinations(range(M), K):
        idx = []
        for h in subset:
            idx += [3*h, 3*h+1, 3*h+2]
        coeff = LeastSquaresSolution(C, R, idx)
        v = sum(coeff[i] * C[:, idx[i]] for i in range(3*K))
        val = NormalizedInnerProduct(v, R, R_norm)
        inners.append(val)
        sol = Solution(val, [s+1 for s in subset], coeff)
        if heapq.nsmallest(1, bestSolutions)[0].inner < sol.inner:
            heapq.heapreplace(bestSolutions, sol)
    end = time.time()

    sorted_inners = sorted(inners, reverse=True)[:best]
    x_sorted = list(range(1, best+1))
    plt.figure()
    plt.scatter(x_sorted, sorted_inners)
    plt.title(f"k={K} (Top {best} Sorted Decreasing)", fontsize=30)
    plt.xlabel("Sorted Candidate Index (1–10)", fontsize=30)
    plt.ylabel("Normalized Inner Product", fontsize=30)

    ax = plt.gca()
    ticks = ax.get_yticks()
    if len(ticks) > 1:
        spacing = abs(ticks[1] - ticks[0])
        decimal_places = max(0, int(-floor(log10(spacing))))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{decimal_places}f'))
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)

    plt.show()

    top_sols = heapq.nlargest(best, bestSolutions)
    PrintConsole(top_sols, str(end - start))
    return ceil(max(abs(sol.weights).max() for sol in top_sols))


def PracticalAlgorithm(C, R, Kmin, Kmax, Kstar=3):
    bigM = 0
    for k in range(1, min(Kmax, Kstar) + 1):
        tmp = EnumerationApproach(C, R, k)
        bigM = max(bigM, tmp)
    print("***\nFrom preliminary experiments, bigM is chosen as", bigM, "***\n")

    for k in range(Kmin, Kmax + 1):
        OptimizationApproach(C, R, k, bigM)


datasets = {4: ['apo_inv_hessian.dat', 'diffE.dat']}
Kmin, Kmax = 1, 3
for key in datasets:
    C = loadtxt(datasets[key][0])
    R = loadtxt(datasets[key][1])
    PracticalAlgorithm(C, R, Kmin, Kmax)

