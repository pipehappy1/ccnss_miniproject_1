import numpy as np
from scipy.ndimage import filters


def ddm_pdf(params, data):
    mu = params[0]
    sigma = params[1]
    ddm1 = DDM(mu, sigma)

    rt_pdf = ddm1.pdf_by_sampling()

    rt_pdf[]
    
    

class DDM:
    def __init__(self, mu=0.0015, sigma=0.05):
        self.bias = 0
        self.mu = mu
        self.sigma = sigma
        self.Bp = 1
        self.Bn = -1

    def unconstrained_sample(self, trails, sample_length=2000):
        all_random = np.random.normal(self.mu, self.sigma, (trails, sample_length))

        all_increment = np.dot(all_random, np.triu(np.ones((sample_length, sample_length))))

        all_increment += self.bias
        return all_increment

    def get_rt(self, unconstrained_samples):

        rt_p = np.argmax(unconstrained_samples > self.Bp, axis=1)
        rt_n = np.argmax(unconstrained_samples < self.Bn, axis=1)

        trail_p = (rt_n > 0)*(rt_p > 0)*(rt_p < rt_n) + (rt_n == 0)*(rt_p > 0)
        trail_n = (rt_p > 0)*(rt_n > 0)*(rt_n < rt_p) + (rt_p == 0)*(rt_n > 0)

        rt_p_only = np.extract(trail_p, rt_p)
        rt_n_only = np.extract(trail_n, rt_n)

        return rt_p, rt_n, trail_p, trail_n, rt_p_only, rt_n_only

    def get_both_rt(self, unconstrained_samples):
        rt_p, rt_n, trail_p, trail_n, rt_p_only, rt_n_only = self.get_rt(unconstrained_samples)

        return rt_p*trail_p + rt_n*trail_n

    def pdf_by_sampling(self, trails=15000, sample_length=400):
        all_increment = self.unconstrained_sample(trails)
        rt = self.get_both_rt(all_increment)

        rt_pdf = np.histogram(rt, bins = sample_length, density=True)[0]

        smooth_rt = filters.gaussian_filter1d(rt_pdf, 7)

        smooth_rt = smooth_rt / np.sum(smooth_rt)

        return smooth_rt
