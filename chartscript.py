import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook

#fname = cbook.get_sample_data('C:\\Users\\Porkenstein\\Documents\\GitHub\\PyBrainEurogame\\ga_ann_lvl4buildings\\performanceovertime_2.csv', asfileobj=False)
#fname = cbook.get_sample_data('C:\\Users\\Porkenstein\\Documents\\GitHub\\PyBrainEurogame\\rl_mcs\\performanceovertime_onehundredthepsilon.csv', asfileobj=False)

EPSILONS = ["e = 1.0", "e = 0.0", "e = 0.001", "e = 0.01", "e =0.1"]

data = np.genfromtxt('C:\\Users\\Porkenstein\\Documents\\GitHub\\PyBrainEurogame\\rl_mcs\\performanceovertime.csv',delimiter=',', dtype = int)
x = data[1:,0]
y = []

for i in range(1, 6):
    y.append(data[1:,i])

for i in range(0, 5):
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y[i], 1))(np.unique(x)), label = EPSILONS[i])
#plt.plotfile(fname, range(0, 1), subplots=False)
plt.axes().legend()

plt.ylim([10,15])
plt.show()