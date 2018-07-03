import numpy as np
import pandas as pd
from scipy import optimize

import mini1.ddm_sampler as ddm

data_file = './data/dots_psychophysics.txt'
tdata = pd.read_csv(data_file, delimiter=' ', skipinitialspace=True, index_col=None, header=None, names=['coherence', 'direction', 'choice', 'rewarded', 'rt'])
tdata = tdata.values
scalar_rt = tdata[:,4]/10
tdata[:,4] = scalar_rt.astype(np.int)
coherence_values = np.unique(tdata[:,0])

#plt.hist(np.extract(tdata[:,0]==0.724, tdata[:, 4]), bins=100);plt.show()

res = optimize.minimize(ddm.ddm_pdf, np.array([0.0015, 0.03]), args=(tdata[:,4].astype(np.int),))
print(res.x)
print(res.success)
print(res.message)

