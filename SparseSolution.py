import time
from numpy import ceil, loadtxt, linalg, inner, matmul, zeros, log10, floor
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

def PrintToFile(message):
    print(message)
    with open("results.txt", "a") as f:
        f.write(message + "\n")
        f.flush()

def PrintConsole(bestSolutions, timeElapsed):
    lines = ["Time Elapsed in seconds: " + str(timeElapsed)]
    for i, sol in enumerate(bestSolutions, 1):
        lines.append(f"Best solution number {i}")
        lines.append("Value: " + str(sol.inner))
        lines.append("Residue Indices (starting from 1): " + str(sol.subset))
        for j in range(len(sol.subset)):
            w = sol.weights[3*j : 3*j+3]
            lines.append(f"{w[0]},{w[1]},{w[2]}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    with open("results.txt", "a") as f:
        f.write(txt + "\n")

def NormalizedInnerProduct(v, R, Rn = None):
    if Rn is None:
        Rn = linalg.norm(R)
    return inner(v, R) / (linalg.norm(v) * Rn)

def OutputModel(start, end, M, z, y, R, regularization):
    sol, subset, weights = zeros(len(R)), [], []
    for i in range(M):
        if z[i].x > 1e-3:
            subset.append(i + 1)
            for h in range(3):
                j = 3*i + h
                weights.append(y[j].x)
                sol += y[j].x * C[:, j]
    PrintConsole([Solution(NormalizedInnerProduct(sol, R), subset, weights)], end - start)

def OutputModelToFile(start, end, M, z, y, R, regularization):
    sol, maxIdx, weights = 0, [], []
    for i in range(M):
        if z[i].x > 1.0e-3:
            maxIdx.append(i + 1)
            w = []
            for h in range(3):
                j = 3 * i + h
                w.append(y[j].x)
                sol += y[j].x * C[:, j]
            weights.append(w)
    ni  = NormalizedInnerProduct(sol, R)
    PrintToFile("Maximum: " + str(ni))
    PrintToFile("Time Elapsed in seconds: " + str(end - start))
    PrintToFile("Maximizer Residue Indices (starting from 1): " + str(maxIdx))
    for w in weights:
        PrintToFile(",".join(map(str, w)))
    PrintToFile("")
    return ni, end - start, maxIdx, weights

def OptimizationApproach(C, R, K, bigM = 100, regularization = -1):
    PrintToFile("Using the OPTIMIZATION approach with..")
    if regularization < 0:
        PrintToFile("k=" + str(K))
        PrintToFile("bigM=" + str(bigM))
    else:
        PrintToFile("Penalty=" + str(regularization))
    start = time.time()
    N, M = len(R), len(R) // 3
    model = Model('normalizedInnerProduct')
    model.setParam('OutputFlag', False)
    y = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    u = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    model.addConstrs(u[i] - quicksum(C[i, j] * y[j] for j in range(N)) == -R[i] for i in range(N))
    obj = quicksum(u[i] * u[i] for i in range(N))
    if regularization < 0:
        z = model.addVars(M, vtype=GRB.BINARY)
        model.addConstrs(y[i] >= -bigM * z[i // 3] for i in range(N))
        model.addConstrs(y[i] <=  bigM * z[i // 3] for i in range(N))
        model.addConstr(z.sum() == K)
        model.setParam('MIPGap', 1e-3)
    else:
        z = model.addVars(M, lb = 0)
        model.addConstrs( y[i] <= z[i // 3] for i in range(N))
        model.addConstrs(-y[i] <= z[i // 3] for i in range(N))
        obj += regularization * z.sum()
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    OutputModelToFile(start, time.time(), M, z, y, R, regularization)

def LeastSquaresSolution(C, R, idx):
    return linalg.solve(matmul(C[idx, :], C[:, idx]), matmul(C[idx, :], R))

def EnumerationApproach(C, R, K, best = 10):
    print("Using the ENUMERATION approach with k=", K)
    start = time.time()
    N, M = len(R), len(R) // 3
    bestS = [Solution(0, [], [])]*best
    Rn, inners = linalg.norm(R), []
    for subset in combinations(range(M), K):
        idx = [3*h + d for h in subset for d in range(3)]
        coeff = LeastSquaresSolution(C, R, idx)
        v = sum(coeff[i] * C[:, idx[i]] for i in range(3*K))
        val = NormalizedInnerProduct(v, R, Rn)
        inners.append(val)
        sol = Solution(val, [s+1 for s in subset], coeff)
        if heapq.nsmallest(1,bestS)[0].inner < sol.inner:
            heapq.heapreplace(bestS, sol)
    end = time.time()
    x = list(range(1, best+1))
    plt.scatter(x, sorted(inners, reverse=True)[:best])
    plt.title(f"k={K} (Top {best} Sorted Decreasing)", fontsize = 30)
    plt.xlabel("Sorted Candidate Index (1–10)", fontsize = 30)
    plt.ylabel("Normalized Inner Product", fontsize = 30)
    ax = plt.gca()
    ticks = ax.get_yticks()
    if len(ticks) > 1:
        dp = max(0, int(-floor(log10(abs(ticks[1]-ticks[0])))))
        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{dp}f'))
    ax.tick_params(axis='x', labelsize = 26)
    ax.tick_params(axis='y', labelsize = 26)
    plt.show()
    PrintConsole(heapq.nlargest(best, bestS), end - start)
    return ceil(max(abs(sol.weights).max() for sol in heapq.nlargest(best, bestS)))

def PracticalAlgorithm(C, R, Kmin, Kmax, Kstar = 3):
    bigM = 0
    for k in range(1, min(Kmax, Kstar) + 1):
        bigM = max(bigM, EnumerationApproach(C, R, k))
    print("***\nFrom preliminary experiments, bigM is chosen as", bigM, "***\n")
    for k in range(Kmin, Kmax + 1):
        OptimizationApproach(C, R, k, bigM)

datasets = {4: ['apo_inv_hessian.dat', 'diffE.dat']}
Kmin, Kmax = 1, 2
for key in datasets:
    C = loadtxt(datasets[key][0])
    R = loadtxt(datasets[key][1])
    PracticalAlgorithm(C, R, Kmin, Kmax)

