### Plausible Selectivity Curve Stuff
### Use plausible selectivity curve
### Create data
### Fit to simple selectivity curves

import matplotlib.pyplot as plt
import random as rand
import math
import scipy.optimize
import statistics as stats
import numpy as np

'''Parameters'''
#selectCurve is the population selectivity density function
def selectCurve(x):
    return 300*(0.11-0.9*x+2.21*x**2-1.4*x**3)/14

#evaluate for many different return rates, given in rVals
rNum = 20
rVals = [(1+i)/rNum for i in range(rNum)]

#data from Mille Lacs Lake
n = 2084
catchTrue = 125000
catchSD = 10000
popTrue = 67000
popSD = 5000
qTrue = catchTrue/popTrue
catch = int(n*qTrue)

#for each return rate, run this many simulations, each time creating
#a catch history and computing the recycling rate estimates
iters = 1000

'''Run simulations'''

#These lists hold the all the results from the simulations
#the true recycling rates, the estimated recycling rates, 
#and the fitted values of p
recRates = []
equalRRs = []
uniformRRs =[]
ziEqualRRs = []
pvalsEqual = []
ziUniformRRs = []
pvalsUniform = []

#run the simluation
for r in rVals:
    #print(r)
    
    recRatesTemp = []
    equalRRsTemp = []
    uniformRRsTemp = []
    ziEqualRRsTemp = []
    pvalsEqualTemp = []
    ziUniformRRsTemp = []
    pvalsUniformTemp = []
    
    for _ in range(iters):
        # Fish population selectivity
        # sample each fish's selectivity from selectCurve
        catchability = []
        while len(catchability)<n:
            x = rand.random()
            if rand.random()<selectCurve(x):
                catchability.append(x)
                
        catchsum = sum(catchability)
                
        # Catch fish until until we have the predetermined number
        # these lists keep track of how many times each fish was
        # caught and recorded
        catches = [0 for _ in range(n)]
        reportedCatches = [0 for _ in range(n)]
        
        for _ in range(catch):
            # first select fish to be caught proportional to selectivity
            val = rand.random()*catchsum
            total = 0
            indx = -1
            while total<val:
                indx += 1
                total += catchability[indx]
            
            # increases catch count
            catches[indx]+=1    
            # with probability r, increase report count
            if rand.random()<r:
                reportedCatches[indx]+=1
                
        # count how many fish were caught i times
        counts = [0 for _ in range(20)]
        for indx in range(20):
            counts[indx] = catches.count(indx)/n
        
        ''' Compute Reporting rates '''
        # True recycling rate comes from the number of fish
        # that were caught zero times
        recRatesTemp.append(qTrue/(1-counts[0]))
        
        # Simulate measuring the catch and population of the lake
        # these values, not the true values, are used to 
        # estimate the recycling rate according to our models
        
        # the equal selectivity model
        catchEst = np.random.normal(catchTrue,catchSD)
        popEst = np.random.normal(popTrue, popSD)
        qEst = catchEst/popEst       
        equalRRsTemp.append(qEst/(1-math.exp(-qEst)))
        
        
        # the uniform selectivity model
        catchEst = np.random.normal(catchTrue,catchSD)
        popEst = np.random.normal(popTrue, popSD)
        qEst = catchEst/popEst  
        uniformRRsTemp.append(2*qEst**2/(2*qEst-1+math.exp(-2*qEst)))
        
        # the zero-inflated equal selectivity model
        # need to fit tag return data to detemine p
        catchEst = np.random.normal(catchTrue,catchSD)
        popEst = np.random.normal(popTrue, popSD)
        qEst = catchEst/popEst 
        def equalLS(p):
            sum=(counts[0] -  (p*math.exp(-r*qEst/p) + (1-p)))**2
            for indx in range(19):
                indx += 1
                sum += (counts[indx]-(p*math.exp(-r*qEst/p)*(r*qEst/p)**indx)/math.factorial(indx))**2
            return sum    
        y1 = scipy.optimize.minimize(equalLS,0.01,bounds=[(0,1)])
        p1 = y1.x[0]
        pvalsEqualTemp.append(p1)
        ziEqualRRsTemp.append(qEst/p1/(1-math.exp(-qEst/p1)))
        
        # the zero-inflated uniform selectivity model
        # need to fit tag return data to determine p
        catchEst = np.random.normal(catchTrue,catchSD)
        popEst = np.random.normal(popTrue, popSD)
        qEst = catchEst/popEst 
        def uniformLS(p):
            sum=(counts[0] - ((1-p) + p*scipy.special.gammainc(1,2*r*qEst)/(2*r*qEst)))**2
            for indx in range(19):
                indx += 1
                sum += (counts[0] - (p*scipy.special.gammainc(indx+1,2*r*qEst)/(2*r*qEst*math.factorial(indx))))**2
            return sum
        y2 = scipy.optimize.minimize(uniformLS,0.01,bounds=[(0,1)])
        p2 = y2.x[0]
        pvalsUniformTemp.append(p2)
        ziUniformRRsTemp.append((2*qEst*qEst/p2)/(2*qEst/p2 - 1 + math.exp(-2*qEst/p2))) 
        
    # store all the values before moving to the next tag return rate
    recRates.append(recRatesTemp)
    equalRRs.append(equalRRsTemp)
    uniformRRs.append(uniformRRsTemp)
    ziEqualRRs.append(ziEqualRRsTemp)
    pvalsEqual.append(pvalsEqualTemp)
    ziUniformRRs.append(ziUniformRRsTemp)
    pvalsUniform.append(pvalsUniformTemp)
    
    
''' Process and plot data '''


#compute the mean for each estimate of recycling rate and
#each tag return rate
recRatesMeans = [stats.mean(x) for x in recRates]
equalRRsMeans = [stats.mean(x) for x in equalRRs]
uniformRRsMeans = [stats.mean(x) for x in uniformRRs]
ziEqualRRsMeans = [stats.mean(x) for x in ziEqualRRs]
ziUniformRRsMeans = [stats.mean(x) for x in ziUniformRRs]

#plot the mean recycling rate for each model and each return rate
plt.subplots()
plt.plot(rVals,recRatesMeans,'b')
plt.plot(rVals,equalRRsMeans,'g')
plt.plot(rVals,uniformRRsMeans,'r')
plt.plot(rVals,ziEqualRRsMeans,'c')
plt.plot(rVals,ziUniformRRsMeans,'m')
plt.xlabel('$\lambda$')
plt.ylabel('Recycling Rate')
plt.legend(['True Recycling Rate','Equal Selectivity','Uniform Selectivity','Zero-inflated Equal Selectivity','Zero-inflated Uniform Selectivity'],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.ylim((2,2.8))
plt.savefig("rrsNoConInt.png", dpi=600, bbox_inches = "tight")


#now identify the 95% confidence interval for the data and plot that
for x in recRates:
    x.sort()
for x in equalRRs:
    x.sort()
for x in uniformRRs:
    x.sort()
for x in ziEqualRRs:
    x.sort()
for x in ziUniformRRs:
    x.sort()

trueUpper = [x[int(iters*0.975)+1] for x in recRates]
trueLower = [x[int(iters*0.025)] for x in recRates]
equalUpper = [x[int(iters*0.975)+1] for x in equalRRs]
equalLower = [x[int(iters*0.025)] for x in equalRRs]
uniformUpper = [x[int(iters*0.975)+1] for x in uniformRRs]
uniformLower = [x[int(iters*0.025)] for x in uniformRRs]
ziEqualUpper = [x[int(iters*0.975)+1] for x in ziEqualRRs]
ziEqualLower = [x[int(iters*0.025)] for x in ziEqualRRs]
ziUniformUpper = [x[int(iters*0.975)+1] for x in ziUniformRRs]
ziUniformLower = [x[int(iters*0.025)] for x in ziUniformRRs]

plt.subplots()
plt.plot(rVals,recRatesMeans,'b')
plt.plot(rVals,equalRRsMeans,'g')
plt.plot(rVals,uniformRRsMeans,'r')
plt.plot(rVals,ziEqualRRsMeans,'c')
plt.plot(rVals,ziUniformRRsMeans,'m')
plt.xlabel('r Value')
plt.ylabel('Recycling Rate')
plt.legend(['True Recycling Rate','Equal Selectivity','Uniform Selectivity','Zero-inflated Equal Selectivity','Zero-inflated Uniform Selectivity'],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.fill_between(rVals, trueLower, trueUpper, color='b', alpha=0.3)
plt.fill_between(rVals, equalLower, equalUpper, color='g', alpha=0.3)
plt.fill_between(rVals, uniformLower, uniformUpper, color='r', alpha=0.3)
plt.fill_between(rVals, ziEqualLower, ziEqualUpper, color='c', alpha=0.3)
plt.fill_between(rVals, ziUniformLower, ziUniformUpper, color='m', alpha=0.3)
plt.savefig("rrsConInt.png", dpi=600, bbox_inches = "tight")


# Plotting the estimate of p for equal and uniform zero-inflated models
# includes confidence intervals
pvalsEqualMeans = [stats.mean(x) for x in pvalsEqual]
pvalsUniformMeans = [stats.mean(x) for x in pvalsUniform]
pvalsEqualSDs = [np.std(x) for x in pvalsEqual]
pvalsUniformSDs = [np.std(x) for x in pvalsUniform]

for x in pvalsEqual:
    x.sort()
for x in pvalsUniform:
    x.sort()

pvalsEqualUpper = [x[int(iters*0.975)+1] for x in pvalsEqual]
pvalsEqualLower = [x[int(iters*0.025)] for x in pvalsEqual]
pvalsUniformUpper = [x[int(iters*0.975)+1] for x in pvalsUniform]
pvalsUniformLower = [x[int(iters*0.025)] for x in pvalsUniform]

plt.subplots()
plt.plot(rVals,pvalsEqualMeans,'b')
plt.plot(rVals,pvalsUniformMeans,'r')
plt.fill_between(rVals, pvalsEqualUpper, pvalsEqualLower, color='b', alpha=0.3)
plt.fill_between(rVals, pvalsUniformUpper, pvalsUniformLower, color='r', alpha=0.3)
plt.xlabel('r Value')
plt.ylabel('Estimate of p')
plt.legend(['Zero-inflated Equal Selectivity','Zero-inflated Uniform Selectivity'],bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.savefig("pvals.png",dpi=600, bbox_inches = "tight")


print(recRates)
print(equalRRs)
print(uniformRRs)
print(ziEqualRRs)
print(ziUniformRRs)


# plot the plausible selectivity curve
svals = [i/1000 for i in range(1001)]
hs = [selectCurve(s) for s in svals]
plt.subplots()
plt.plot(svals,hs)
plt.xlabel('s')
plt.ylabel('Population Density $h(s)$')
plt.savefig("selectivityCurve.png",dpi=600)
