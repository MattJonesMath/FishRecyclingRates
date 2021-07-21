### 
# Fishing
# Create synthetic data for fishing model
# Set fish pop size, total catch, reporting rate, frac catchable
# Gaussian distribution of catchabilities
###

import matplotlib.pyplot as plt
import random as rand
import math
import scipy.optimize
import statistics as stats
import numpy as np
import scipy.integrate


n = 2100
catch = 125000/70000
r=1
sigma = 0.05

def gauss(x,s):
    return math.exp(-0.5*((x-0.5)/s)**2)/(s*math.sqrt(2*math.pi))
        

catches = [0 for _ in range(n)]
catchability = []
while len(catchability)<n:
    c = rand.random()
    samplerand = rand.random()
    if samplerand<gauss(c,sigma):
        catchability.append(c)

catchsum = sum(catchability)

# plt.subplots()
# plt.hist(catchability)


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

def integrand(x, indx, sigma):
    return math.exp(-x*2*catch)*(x**indx)*gauss(x,sigma)

for indx in range(20):
    analytic[indx] = ((2*catch)**indx)*(scipy.integrate.quad(integrand,0,1,args=(indx,sigma))[0])/math.factorial(indx)




plt.subplots()
plt.plot(counts)
plt.plot(analytic)


def ls(sig):
    sum = 0
    
    def integrand(x,indx):
        return math.exp(-x*2*catch)*(x**indx)*gauss(x,sig)
    
    for indx in range(20):
        sum += (counts[indx] - ((2*catch)**indx)*(scipy.integrate.quad(integrand,0,1,args=(indx))[0])/math.factorial(indx))**2
    return sum

y = scipy.optimize.minimize(ls,0.01)
print(y)