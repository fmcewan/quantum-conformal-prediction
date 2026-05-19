from scipy.stats import norm

class Normal:
   
    def __init__(self, loc=0, scale=1):
        self._dist = norm(loc=loc, scale=scale)
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        return self._dist.cdf(x)

    def pdf(self, x):
        return self._dist.pdf(x)

    def rvs(self, size=1):
        return self._dist.rvs(size=size)

    def get_hdr(self, mass):
        return [self._dist.interval(mass)]
