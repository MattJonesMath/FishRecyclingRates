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
import json


n = 2100
# catch = int(n*125000/70000)
p = 0.2

iters = 1000
rvalnum = 100

rvals = [(i+1)/rvalnum for i in range(rvalnum)]
pvals = []

for r in rvals:
    
    if (r*10)%1==0:
        print(r)
        
    pvalstemp = []
    
    for _ in range(iters):
        
        catch = int(n*np.random.normal(125000,10000)/np.random.normal(70000,5000))

        catches = [0 for _ in range(n)]
        
        # Catch fish
        for i in range(catch):
            indx = rand.randrange(int(n*p))
            if rand.random()<r:    
                catches[indx]+=1
                 
        counts = [0 for _ in range(20)]
        
        for indx in range(20):
            counts[indx] = catches.count(indx)/n
        
        
        def ls(p):
            sum=(counts[0] -  (p*math.exp(-r*catch/(p*n)) + (1-p)))**2
            for indx in range(19):
                indx += 1
                sum += (counts[indx]-p*math.exp(-r*catch/(p*n))*((r*catch/(p*n))**indx)/math.factorial(indx))**2
            return sum
        
        #y = scipy.optimize.minimize(ls,0.01,bounds=[(0,1)])
        y = scipy.optimize.minimize(ls,0.01)
        pvalstemp.append(y.x[0])
        
    pvals.append(pvalstemp)
    
    
stdevs = [np.std(pvals[i]) for i in range(len(pvals))]

plt.subplots()
plt.plot(rvals,stdevs)
plt.plot(rvals, [stats.mean(x) for x in pvals])




# Save data

with open('pvals02popuncertainty.json', 'w') as f:
    json.dump(pvals, f)

