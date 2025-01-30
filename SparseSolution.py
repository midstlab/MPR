
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:18:49 2024

@author: Burak
"""
print("Script execution started.")

import time
from numpy import ceil, loadtxt, linalg, inner, matmul, zeros
from gurobipy import GRB, Model, quicksum
from itertools import combinations

# Open a log file to store outputs
output_file = open("optimization_output.txt", "a")

def PrintToFile(message):
    print(message)
    output_file.write(message + "\n")
    output_file.flush()  # Ensure the message is written immediately

def Output(m):  
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
    status = m.status
    PrintToFile('The optimization status is ' + status_code[status])
    if status == 2:    
        PrintToFile('Optimal solution:')
        for v in m.getVars():
            if abs(v.x) > 10e-5:
                PrintToFile(str(v.varName) + " = " + str(round(v.x, 5)))    
        PrintToFile('Optimal objective value: ' + str(m.objVal) + "\n")

def Norm2Sq(v):
    N = len(v)
    return sum(v[i]**2 for i in range(N))

def LinearSolve(C, R): # solve Cf = R
    start = time.time()
    f = linalg.solve(C, R)
    end = time.time()
    PrintToFile("Time Elapsed in seconds (Linear Solve): " + str(end - start))
    return f

def OutputModelToFile(start, end, M, z, y, R, regularization):
    sol = 0
    maxIndices = []
    weights = []
    
    for i in range(M):
        if z[i].x > 1.0e-3:
            maxIndices.append(i + 1)
            weight = []
            for h in range(3):
                j = 3 * i + h
                weight.append(y[j].x)
                sol += y[j].x * C[:, j]
            weights.append(weight)
    
    norm_inner = NormalizedInnerProduct(sol, R)
    elapsed_time = str(end - start)
    
    # Write output to file and console
    PrintToFile("Maximum: " + str(norm_inner))
    PrintToFile("Time Elapsed in seconds: " + elapsed_time)
    PrintToFile("Maximizer Residue Indices (starting from 1): " + str(maxIndices))
    for i in range(len(maxIndices)):
        PrintToFile(",".join(map(str, weights[i])))
    PrintToFile("")
    
    return norm_inner, elapsed_time, maxIndices, weights

def OptimizationApproach(C, R, K, bigM=100, regularization=-1):
    PrintToFile("Using the OPTIMIZATION approach with..")
    if regularization < 0: 
        PrintToFile("k=" + str(K))
        PrintToFile("bigM=" + str(bigM))
    if regularization >= 0:
        PrintToFile("Penalty=" + str(regularization))

    start = time.time()
    N = len(R)
    M = int(N/3)
    
    model = Model('normalizedInnerProduct')
    model.setParam('OutputFlag', False)
    
    # Add variables
    y = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='y')
    u = model.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='u')
    model.addConstrs(u[i] - quicksum(C[i, j] * y[j] for j in range(N)) == -R[i] for i in range(N))
    
    objFunction = quicksum(u[i] * u[i] for i in range(N))
        
    if regularization < 0.0:
        z = model.addVars(M, vtype=GRB.BINARY, name='z') 
        model.addConstrs(y[i] >= -bigM * z[int(i/3)] for i in range(N))
        model.addConstrs(y[i] <= bigM * z[int(i/3)] for i in range(N))
        model.addConstr(z.sum('*') == K)
        model.setParam('MIPGap', 1e-3)  
    else:
        z = model.addVars(M, lb=0, ub=GRB.INFINITY, name='z')  
        model.addConstrs(y[i] <= z[int(i/3)] for i in range(N))
        model.addConstrs(-y[i] <= z[int(i/3)] for i in range(N))
        objFunction = objFunction + regularization * z.sum('*')
        
    model.setObjective(objFunction, GRB.MINIMIZE)
    model.optimize()
    end = time.time()
    OutputModelToFile(start, end, M, z, y, R, regularization)

def LeastSquaresSolution(C, R, indices):
    return linalg.solve(matmul(C[indices, :], C[:, indices]), matmul(C[indices, :], R))

def NormalizedInnerProduct(v, R, R_norm=-1):
    if R_norm < 0:
        R_norm = linalg.norm(R, 2)
    return inner(v, R) / (linalg.norm(v, 2) * R_norm)

def EnumerationApproach(C, R, K):
    PrintToFile("Using the ENUMERATION approach with..")
    PrintToFile("k=" + str(K))

    start = time.time()
    N = len(R)
    M = int(N/3)
    maxInner = -1
    maxInnerIndex = zeros(K)
    maxWeights = zeros(3*K)
    R_norm = linalg.norm(R, 2)
    
    allSubsets = list(combinations(range(M), K))
    for subset in allSubsets:
        indices = []
        for h in subset:
            indices += [3 * h, 3 * h + 1, 3 * h + 2]
        coeff = LeastSquaresSolution(C, R, indices)
        v = sum(coeff[i] * C[:, indices[i]] for i in range(3 * K))
        temp = NormalizedInnerProduct(v, R, R_norm)
        if temp > maxInner:
            maxInner = temp
            maxInnerIndex = subset
            maxWeights = coeff.copy()
    end = time.time()
    
    weights = []
    for k in range(K):
        weights.append([maxWeights[3*k], maxWeights[3*k+1], maxWeights[3*k+2]])
    PrintToFile(f"Maximum: {maxInner}")
    PrintToFile(f"Time Elapsed in seconds: {end - start}")
    PrintToFile(f"Maximizer Residue Indices (starting from 1): {[i+1 for i in maxInnerIndex]}")
    for w in weights:
        PrintToFile(f"Weights: {w}")
    PrintToFile("")
    return ceil(max(abs(maxWeights)))

def PracticalAlgorithm(C, R, Kmin, Kmax, Kstar=5):
    bigM = 0
    for k in range(1, min(Kmax, Kstar) + 1):
        temp = EnumerationApproach(C, R, k)
        if temp > bigM:
            bigM = temp
    
    PrintToFile("***")
    PrintToFile("From preliminary experiments, bigM is chosen as " + str(bigM))
    PrintToFile("***\n")
    
    for k in range(Kmin, Kmax + 1):
        OptimizationApproach(C, R, k, bigM)

datasets = {
    4: ['apo_inv_hessian.dat', 'diffE.dat']  
}

Kmin, Kmax = 1, 5
for key in datasets.keys():
    dataset = datasets[key]
    C = loadtxt(dataset[0])  
    R = loadtxt(dataset[1])  
    PracticalAlgorithm(C, R, Kmin, Kmax)



# Close the output file after processing
output_file.close()

for key in datasets.keys():
    dataset = datasets[key]
    print(f"Processing dataset: {dataset}")  # Debug output
    C = loadtxt(dataset[0])  
    print(f"Matrix C loaded with shape {C.shape}")  # Debug output
    R = loadtxt(dataset[1])  
    print(f"Vector R loaded with shape {R.shape}")  # Debug output
    PracticalAlgorithm(C, R, Kmin, Kmax)
