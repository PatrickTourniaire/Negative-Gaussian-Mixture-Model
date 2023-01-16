# External imports

# Local imports
from distributions import IDistribution

class ComputationalUnit():

    def __init__(self, dist: IDistribution):
        self.dist = dist
    
    def likelihood(self, sample):
        return self.dist.likelihood(sample)

    def likelihood_log(self, sample):
        return self.dist.likelihood_log(sample)

    def likelihood_neglog(self, sample):
        return self.dist.likelihood_neglog(sample)
