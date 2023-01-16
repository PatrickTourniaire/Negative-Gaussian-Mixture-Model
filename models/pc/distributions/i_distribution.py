# External imports
from abc import ABC, abstractmethod


class IDistribution(ABC):
    """Interface class for implementing valid PDF distributions
    which are used to encode computational units.
    Args:
        ABC (Class): inherits the abstract class from abc library
    """
    @abstractmethod
    def params(self):
        pass
    
    @abstractmethod
    def likelihood(self):
        pass

    @abstractmethod
    def likelihood_log(self):
        pass
    
    @abstractmethod
    def likelihood_neglog(self):
        pass
