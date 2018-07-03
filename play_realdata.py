import numpy as np
import pandas as pd

data_file = './data/dots_psychophysics.txt'
tdata = pd.read_csv(data_file, delimiter=' ', skipinitialspace=True, index_col=None, header=None, names=['coherence', 'direction', 'choice', 'rewarded', 'rt'])
tdata = tdata.values
scalar_rt = tdata[:,4]/10
tdata[:,4] = scalar_rt.astype(np.int)
coherence_values = np.unique(tdata[:,0])

plt.hist(np.extract(tdata[:,0]==0.724, tdata[:, 4]), bins=100);plt.show()

