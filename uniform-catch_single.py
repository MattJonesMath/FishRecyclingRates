### 
# Fishing
# Create synthetic data for fishing model
# Set fish pop size, total catch, reporting rate, frac catchable
# Uniform distribution of catchabilities
###

import matplotlib.pyplot as plt
import random as rand
import math
import scipy.optimize
import statistics as stats
import numpy as np


n = 2100
catch = 10*125000/70000
r=1
p=0.8
        

catches = [0 for _ in range(n)]

catchability = [0 for _ in range(n)]
for indx in range(int(n*p)):
    catchability[indx] = rand.random()

catchsum = sum(catchability)

# Catch fish
for _ in range(int(n*catch)):
    
    val = rand.random()*catchsum
    total = 0
    indx = -1
    while total<val:
        indx += 1
        total += catchability[indx]
        
    if rand.random()<r:
        catches[indx]+=1
        
        
        
counts = [0 for _ in range(20)]
for indx in range(20):
    counts[indx] = catches.count(indx)/n
    
analytic = [0 for _ in range(20)]

analytic[0]+=(1-p)

for indx in range(20):
    analytic[indx] += p*scipy.special.gammainc(indx+1,r*2*catch)/(r*2*catch)




plt.subplots()
plt.plot(counts)
plt.plot(analytic)


print(catch/(1-counts[0]))