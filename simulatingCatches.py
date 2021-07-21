### 
# Fishing
# Create synthetic data for fishing model
# Set fish pop size, total catch, reporting rate, frac catchable
###

import matplotlib.pyplot as plt
import random as rand
import math
import scipy.optimize
import statistics as stats
import numpy as np


n = 2100
catch = int(n*125000/70000)
p = 0.8

iters = 200
rvalnum = 100

rvals = [(i+1)/rvalnum for i in range(rvalnum)]

pvals = []

for r in rvals:
    
    pvalstemp = []
    
    for _ in range(iters):
        

        catches = [0 for _ in range(n)]
        
        # Catch fish
        for i in range(catch):
            indx = rand.randrange(int(n*p))
            if rand.random()<r:    
                catches[indx]+=1
                 
        counts = [0 for _ in range(20)]
        
        for indx in range(20):
            counts[indx] = catches.count(indx)/n
        analytic = [0 for _ in range(20)]
        for indx in range(20):
            analytic[indx] = p*math.exp(-r*catch/(p*n))*((r*catch/(p*n))**indx)/math.factorial(indx) 
        analytic[0] += (1-p)    
        
        
        
        
        # plt.subplots()
        # plt.plot(counts)
        # plt.plot(analytic)
        
        
        def ls(p):
            sum=(counts[0] -  (p*math.exp(-r*catch/(p*n)) + (1-p)))**2
            for indx in range(19):
                indx += 1
                sum += (counts[indx]-p*math.exp(-r*catch/(p*n))*((r*catch/(p*n))**indx)/math.factorial(indx))**2
            return sum
        
        y = scipy.optimize.minimize(ls,0.01)
        pvalstemp.append(y.x[0])
        #print(y)
        
    pvals.append(pvalstemp)
    
    
stdevs = [np.std(pvals[i]) for i in range(len(pvals))]

plt.subplots()
plt.plot(rvals,stdevs)

