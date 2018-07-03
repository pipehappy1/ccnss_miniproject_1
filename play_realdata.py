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

plt.hist(np.extract(tdata[:,0]==0.724, tdata[:, 4]), bins=100);plt.show()

#res = optimize.minimize(ddm.ddm_pdf, np.array([0.0015, 0.03]), args=(tdata[:,4].astype(np.int),))
#print(res.x)
#print(res.success)
#print(res.message)

res = optimize.minimize(ddm.ddm_pdf, np.array([0.0015, 0.03]), args=(fake_data), method = 'Nelder-Mead', tol=0.05, options={'maxiter':200}); print(res)

# final_simplex: (array([[0.00177656, 0.05034375],
#       [0.00168281, 0.05371875],
#       [0.00159844, 0.04715625]]), array([5.27673636, 5.28378169, 5.29123604]))
#           fun: 5.276736355372437
#       message: 'Optimization terminated successfully.'
#          nfev: 15
#           nit: 7
#        status: 0
#       success: True
#             x: array([0.00177656, 0.05034375])

real_data = tdata[tdata[:,0] == 0.032, 4]
res = optimize.minimize(ddm.ddm_pdf, np.array([0.0015, 0.03]), args=(real_data.astype(np.int)), method = 'Nelder-Mead', tol=0.05, options={'maxiter':200}); print(res)


for i in range(len(coherence_values)-1):
    real_data = tdata[tdata[:,0] == coherence_values[i+1], 4]
    res = optimize.minimize(ddm.ddm_pdf, np.array([0.0015, 0.03]), args=(real_data.astype(np.int)), method = 'Nelder-Mead', tol=0.00005, options={'maxiter':200})
    
