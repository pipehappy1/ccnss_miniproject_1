import matplotlib
matplotlib.use('TkAgg')
import mini1.ddm_sampler as ddm
import matplotlib.pyplot as plt
import numpy as np

ddm1 = ddm.DDM()
all_increment = ddm1.unconstrained_sample(10000)

#plt.plot(np.transpose(all_increment))
#plt.savefig('test1.jpg')

rt_p, rt_n, trail_p, trail_n, rt_p_only, rt_n_only = ddm1.get_rt(all_increment)
rts = ddm1.get_both_rt(all_increment)
plt.subplot(3, 1, 1)
plt.hist(rts, bins=50)
plt.subplot(3, 1, 2)
plt.hist(rt_p_only, bins=50)
plt.subplot(3, 1, 3)
plt.hist(rt_n_only, bins=50)
plt.show()
