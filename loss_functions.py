import numpy as np
from scipy.sparse.csc import csc_matrix
from typing import Union, Any

from regularizers import L2NormSquared


class Logistic:

    def __init__(self, X: Union[np.ndarray, csc_matrix], y: np.ndarray,
                 regularizer: Any = L2NormSquared(), l: float = None):
        """ initialize our Logistic regression function

        :param X: An m x n matrix of sample inputs. x_i is in R^n and x_i^T represents
        the ith row of X (b/c its 1 x n).
        :param y: An m x 1 vector in {-1, 1}^m of sample outputs
        :param regularizer: Which regularizer to use. Currently only 'l_2' norm is available
        :param l: The lambda value to use on the norm. If none, will use 1/m
        """
        assert isinstance(X, csc_matrix) or isinstance(X, np.ndarray), 'X must be a csc_matrix or np.ndarray'
        assert isinstance(y, np.ndarray), 'y must be an np.ndarray'
        assert y.shape == (X.shape[0], 1), 'y must have same number of rows as X but only 1 column'
        assert l is None or l > 0, 'l must be positive'
        for method_name in ['f', 'g']:
            assert hasattr(regularizer, method_name), f'regularizer must have attribute {method_name}'
            assert callable(getattr(regularizer, method_name)), f'regularizer.{method_name} must be function'

        self.m, self.n = X.shape
        self.X = X.toarray()
        self.y = y
        self.regularizer = regularizer
        self.l = l or 1/self.m
        self.f_calls = 0
        self.g_calls = 0

    def f(self, v: np.ndarray) -> float:
        """Evaluate our Logistic Regression function at the hyperplane w^T x = b,
        where w = v[:-1] and b = v[-1]

        :param v: an (n + 1) x 1 vector in which v[:-1] is the coefficients of our
        separating hyperplane and w[-1] is the offset our separating hyperplane has from the origin
        :return: the current total loss of our model with this hypothesis
        """
        assert isinstance(v, np.ndarray) and v.shape == (self.n + 1, 1), 'w must be (n + 1) x 1 np.ndarray'
        self.f_calls += 1

        w, b = v[:-1], v[-1]
        return 1/self.m * np.sum(np.log(1 + np.exp(-self.y * (self.X @ w - b)))) + \
            self.l * self.regularizer.f(w)

    def g(self, v: np.ndarray, row_indices: np.ndarray = None,
          col_indices: np.ndarray = None, u: np.ndarray = None) -> np.ndarray:
        """Evaluate our Logistic Regression gradient at the hyperplane w^T x = b,
        where w = v[:-1] and b = v[-1]. Provide row_indices to calculate a stochastic
        gradient. Provide col_indices and u to calculate coordinate gradient.
        For coordinate gradient, only the gradient of col_indices will be returned.

        :param v: an (n + 1) x 1 vector in which v[:-1] is the coefficients of our
        separating hyperplane and w[-1] is the offset our separating hyperplane has from the origin
        :param row_indices: subset of indices of samples (i.e. subset of rows of X and y)
        used for calculating an approximate gradient (used in stochastic gradient)
        :param col_indices: subset of indices of features (i.e. subset of cols of X)
        used for calculating an approximate gradient (used in coordinate descent)
        :param u: vector of precomputed self.X * v.
        :return: the direction of greatest increase of our current total loss of
        our model with this hypothesis.
        """
        coordinate_descent = False
        assert isinstance(v, np.ndarray) and v.shape == (self.n + 1, 1), 'v must be (n + 1) x 1 np.ndarray'
        if row_indices is not None:
            assert isinstance(row_indices, np.ndarray), 'indices must be an array'
            assert (0 <= row_indices).all() and (row_indices <= self.m - 1).all(), \
                'indices must be between 0 and m-1'
        assert not ((col_indices is None) ^ (u is None)), 'both must be None or neither'
        if col_indices is not None and u is not None:
            coordinate_descent = True
            assert isinstance(col_indices, np.ndarray), 'indices must be an array'
            assert (0 <= col_indices).all() and (col_indices <= self.n - 1).all(), \
                'indices must be between 0 and n-1'
            assert isinstance(u, np.ndarray) and u.shape == (self.m, 1), 'u must be m x 1 np.ndarray'

        self.g_calls += 1

        # select subset of indices if stochastic descent
        X = self.X[row_indices] if row_indices is not None else self.X
        y = self.y[row_indices] if row_indices is not None else self.y
        m = y.shape[1] if row_indices is not None else self.m

        if coordinate_descent:
            # select subset of indices if coordinate descent
            X = X[:, col_indices]
            v = v[col_indices]
            v_prime = self.l * self.regularizer.g(v) + \
                1 / m * np.sum((-y * X) / (np.exp(y * u) + 1), axis=0, keepdims=True).T

        else:
            w, b = v[:-1], v[-1]
            # note: sigmoid is 1/(np.exp(self.y * (self.X @ w - b) + 1)
            w_prime = self.l * self.regularizer.g(w) + \
                1/m * np.sum((-y * X)/(np.exp(y * (X @ w - b)) + 1), axis=0, keepdims=True).T

            b_prime = 1/m * np.sum((-y)/(np.exp(y * (X @ w - b)) + 1), keepdims=True)
            v_prime = np.append(w_prime, b_prime, axis=0)

        return v_prime

    def zero_one_loss(self, v: np.ndarray) -> float:
        """Check the zero-one loss (i.e. accuracy) of the hyperplane w^T x = b,
        where w = v[:-1] and b = v[-1]

        :param v: an (n + 1) x 1 vector in which v[:-1] is the coefficients of our
        separating hyperplane and w[-1] is the offset our separating hyperplane has from the origin
        :return: proportion of samples in X that the hyperplane w^T x = b correctly classifies
        """
        assert isinstance(v, np.ndarray) and v.shape == (self.n + 1, 1), 'w must be (n + 1) x 1 np.ndarray'

        w, b = v[:-1], v[-1]
        predictions = -self.y * (self.X @ w - b)
        right_predictions = np.sum(predictions < 0)
        return right_predictions / self.m
