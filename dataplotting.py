###############
# Data fitting plots
# Use data from catchable-uncatchable_varyr.py
###############

import matplotlib.pyplot as plt
import json
import statistics as stats
import numpy as np

rvalnum = 100
rvals = [(i+1)/rvalnum for i in range(rvalnum)]

with open("pvals08popuncertainty.json") as f:
    pvals08 = json.load(f)

means08 = [stats.mean(x) for x in pvals08]
stdevs08 = [np.std(x) for x in pvals08]

with open("pvals05popuncertainty.json") as f:
    pvals05 = json.load(f)
    
means05 = [stats.mean(x) for x in pvals05]
stdevs05 = [np.std(x) for x in pvals05]

with open("pvals02popuncertainty.json") as f:
    pvals02 = json.load(f)
    
means02 = [stats.mean(x) for x in pvals02]
stdevs02 = [np.std(x) for x in pvals02]

upper08 = np.array(means08) + 2*np.array(stdevs08)
lower08 = np.array(means08) - 2*np.array(stdevs08)

upper05 = np.array(means05) + 2*np.array(stdevs05)
lower05 = np.array(means05) - 2*np.array(stdevs05)

upper02 = np.array(means02) + 2*np.array(stdevs02)
lower02 = np.array(means02) - 2*np.array(stdevs02)



plt.subplots()
plt.plot(rvals, means08, 'b')
plt.plot(rvals, means05, 'r')
plt.plot(rvals, means02, 'g')

plt.fill_between(rvals, means08, upper08, color='b', alpha=0.3)
plt.fill_between(rvals, means08, lower08, color='b', alpha=0.3)

plt.fill_between(rvals, means05, upper05, color='r', alpha=0.3)
plt.fill_between(rvals, means05, lower05, color='r', alpha=0.3)

plt.fill_between(rvals, means02, upper02, color='g', alpha=0.3)
plt.fill_between(rvals, means02, lower02, color='g', alpha=0.3)

plt.ylim(0,1)

