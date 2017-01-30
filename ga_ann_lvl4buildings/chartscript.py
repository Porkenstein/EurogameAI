import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook

#fname = cbook.get_sample_data('C:\\Users\\Porkenstein\\Documents\\GitHub\\PyBrainEurogame\\ga_ann_lvl4buildings\\performanceovertime_2.csv', asfileobj=False)
fname = cbook.get_sample_data('C:\\Users\\Porkenstein\\Documents\\GitHub\\PyBrainEurogame\\rl_mcs\\performanceovertime_onehundredthepsilon.csv', asfileobj=False)


plt.plotfile(fname, range(0, 11), subplots=False)
plt.axes().legend_.remove()

plt.show()