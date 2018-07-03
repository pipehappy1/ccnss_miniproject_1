import numpy as np

class DDM:
    def __init__(self):
        self.bias = 0
        self.mu = 0.0015
        self.sigma = 0.05
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

        return rt_p, rt_n, trail_p, trail_n, np.extract(trail_p, rt_p), np.extract(trail_n, rt_n)

    def get_both_rt(self, unconstrained_samples):
        rt_p, rt_n, trail_p, trail_n, rt_p_only, rt_n_only = self.get_rt(unconstrained_samples)

        return rt_p*trail_p + rt_n*trail_n

