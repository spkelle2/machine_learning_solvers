import numpy as np


class L2NormSquared:

    @staticmethod
    def f(v: np.ndarray):
        """calculates the l_2 Norm squared of v

        :param v: a vector to take the squared norm of
        :return:
        """
        return np.sum(v**2)

    @staticmethod
    def g(v: np.ndarray):
        """calculates the gradient of the l_2 Norm squared of v

        :param v: a vector to take the gradient of the squared norm of
        :return:
        """
        return 2*v
