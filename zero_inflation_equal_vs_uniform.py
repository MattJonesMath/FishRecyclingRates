##########
# Fishing
# Create zero-inflated catchability populations
# Create populations of equal and distributed catchability
# Fit data to determine zero-inflation fraction
# Compare data of equal and distributed
##########

import matplotlib.pyplot as plt
import random as rand
import math
import scipy.optimize
import statistics as stats
import numpy as np
import json


n = 2100
catch = 125000/70000
catchnum = int(n*catch)
p = 0.5

iters = 100
rvalnum = 10

rvals = [(i+1)/rvalnum for i in range(rvalnum)]

pvalsequal = []
pvalsuniform = []


#####
# equal catchability fits
#####
for r in rvals:
    
    if (r*10)%1==0:
        print(r)
        
    pvalstemp = []
    
    for _ in range(iters):
        

        catches = [0 for _ in range(n)]
        
        # Catch fish
        for i in range(catchnum):
            indx = rand.randrange(int(n*p))
            if rand.random()<r:    
                catches[indx]+=1
                 
        counts = [0 for _ in range(20)]
        
        for indx in range(20):
            counts[indx] = catches.count(indx)/n
        
        
        # def ls(p):
        #     sum=(counts[0] -  (p*math.exp(-r*catch/(p)) + (1-p)))**2
        #     for indx in range(19):
        #         indx += 1
        #         sum += (counts[indx]-p*math.exp(-r*catch/(p))*((r*catch/(p))**indx)/math.factorial(indx))**2
        #     return sum
        
        def ls(p):
            gmax = 2*catch/p
            sum = 0
            sum += (counts[0] - (1-p) - p*scipy.special.gammainc(1,r*gmax)/(r*gmax))**2
            for indx in range(19):
                indx += 1
                sum += (counts[indx] - p*scipy.special.gammainc(indx+1,r*gmax)/(r*gmax*math.factorial(indx)))**2
            return sum
        
        
        #y = scipy.optimize.minimize(ls,0.01,bounds=[(0,1)])
        y = scipy.optimize.minimize(ls,0.01)
        pvalstemp.append(y.x[0])
        
    pvalsequal.append(pvalstemp)
    
    
    
#####
# Uniform distribution fits
#####
for r in rvals:
    
    if (r*10)%1==0:
        print(r)
        
    pvalstemp = []
    
    for _ in range(iters):
        

        catches = [0 for _ in range(n)]
        
                
        # Catch fish
        catchability = [0 for _ in range(n)]
        for indx in range(int(n*p)):
            catchability[indx] = rand.random()
        catchsum = sum(catchability)
        
        for _ in range(catchnum):
            val = rand.random()*catchsum
            total = 0
            indx = -1
            while total<val:
                indx += 1
                total += catchability[indx]
            
                
            if rand.random()<r:
                catches[indx] += 1
                
            
        counts = [0 for _ in range(20)]
        
        for indx in range(20):
            counts[indx] = catches.count(indx)/n
        

        # def ls(p):
        #     gmax = 2*catch/p
        #     sum = 0
        #     sum += (counts[0] - (1-p) - p*scipy.special.gammainc(1,r*gmax)/(r*gmax))**2
        #     for indx in range(19):
        #         indx += 1
        #         sum += (counts[indx] - p*scipy.special.gammainc(indx+1,r*gmax)/(r*gmax*math.factorial(indx)))**2
        #     return sum
        
        def ls(p):
            sum=(counts[0] -  (p*math.exp(-r*catch/(p)) + (1-p)))**2
            for indx in range(19):
                indx += 1
                sum += (counts[indx]-p*math.exp(-r*catch/(p))*((r*catch/(p))**indx)/math.factorial(indx))**2
            return sum
        
        #y = scipy.optimize.minimize(ls,0.01,bounds=[(0,1)])
        y = scipy.optimize.minimize(ls,0.01)
        pvalstemp.append(y.x[0])
        
    pvalsuniform.append(pvalstemp)
    
    
    
stdevsequal = [np.std(x) for x in pvalsequal]
stdevsuniform = [np.std(x) for x in pvalsuniform]

meansequal = [stats.mean(x) for x in pvalsequal]
meansuniform = [stats.mean(x) for x in pvalsuniform]

plt.subplots()
plt.plot(rvals,stdevsequal)
plt.plot(rvals, meansequal)

plt.plot(rvals,stdevsuniform)
plt.plot(rvals, meansuniform)