import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

data_file = './data/dots_psychophysics.txt'
tdata = pd.read_csv(data_file, delimiter=' ', skipinitialspace=True, index_col=None, header=None, names=['coherence', 'direction', 'choice', 'rewarded', 'rt'])
tdata = tdata.values
scalar_rt = tdata[:,4]/10
tdata[:,4] = scalar_rt.astype(np.int)
coherence_values = np.unique(tdata[:,0])

values = []
for i in range(len(coherence_values)):
    values.append(np.sum((tdata[:,0] == coherence_values[i])* (tdata[:,3] == 1)) / np.sum(tdata[:,0] == coherence_values[i]))

values[0] = np.sum((tdata[:,0]== 0)* (tdata[:,2]== 2)) / (np.sum((tdata[:,0]== 0)* (tdata[:,2]== 2)) + np.sum((tdata[:,0]== 0)* (tdata[:,2]== 3)))


plt.plot(values)
plt.show()





