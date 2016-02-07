from abc import ABCMeta, abstractmethod


class PriorForCoefficient:
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_string(self):
        pass


class NormalPrior(PriorForCoefficient):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "normal({0},{1})".format(self.mu, self.sigma)


class StudentTPrior(PriorForCoefficient):
    def __init__(self, nu, mu, sigma):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return self.to_string()

    def to_string(self):
        return "student_t({0},{1},{2})".format(self.nu, self.mu, self.sigma)
