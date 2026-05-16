from scipy.stats import norm
class Normal(norm):
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)

    def get_hdr(self, mass):
        return [self.interval(mass)]