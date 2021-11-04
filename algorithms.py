import numpy as np
import sys

from loss_functions import Logistic


class Algorithm:
    """Base class for building different optimization algorithms to train a machine
    learning model with linear hypotheses
    """

    def __init__(self, training_loss_function: Logistic, v0: np.ndarray = None,
                 max_iterations: int = 100000, tolerance: float = .001,
                 a_init: float = 1, c: float = .001, eta: float = .5,
                 testing_loss_function: Logistic = None,
                 track_errors: bool = False) -> None:
        """ Instantiation

            :param training_loss_function: Which loss function to use for training
            :param v0: Which hypothesis to start with
            :param max_iterations: After how many iterations to quit if gradient of loss
            function not within tolerance
            :param tolerance: Once the norm of the gradient of the loss function is less than
            this value, we will return the current hypothesis as optimal
            :param a_init: When conducting back tracking Armijo line search,
            what proportion of the initial step to take
            :param c: When conducting back tracking Armijo line search, minimum
            relative improvement our step must take
            :param eta: When conducting back tracking Armijo line search, fraction
            to multiply current step by if it does not meet minimum improvement
            :param testing_loss_function: Which loss function to use for testing
            :param track_errors: Whether or not to track testing and training error as we train
        """

        # sanity checks
        assert isinstance(training_loss_function, Logistic), 'must use instance of logistic loss function'
        if v0 is None:
            v0 = np.zeros((training_loss_function.n + 1, 1))
        else:
            assert isinstance(v0, np.ndarray), 'starting point should be ndarray'
        assert max_iterations > 0 and isinstance(max_iterations, int), 'iterations are a positive integer'
        assert 0 < tolerance, 'tolerance must be positive'
        assert a_init > 0
        assert c > 0
        assert 0 < eta < 1, 'we need a proportion that will shrink our step each line search iteration'
        if testing_loss_function is not None:
            assert isinstance(testing_loss_function, Logistic), 'must use instance of logistic loss function'
        assert isinstance(track_errors, bool)

        # instantiate
        self.training_loss_function = training_loss_function
        self.iteration = 0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.v = v0
        self.v_prev = self.v
        self.f = self.training_loss_function.f(self.v)
        self.g = self.evaluate_gradient()
        self.norm_v = np.linalg.norm(self.v)
        self.norm_vprev = self.norm_v
        self.norm_g = np.linalg.norm(self.g)
        self.a_init = a_init
        self.c = c
        self.eta = eta
        self.testing_loss_function = testing_loss_function
        self.track_errors = track_errors

        # values that get assigned at run time
        self.p = None
        self.a = None
        self.status = 'unsolved'
        self.norm_p = None
        self.training_error = {}
        self.testing_error = {}

    def solve(self):
        """ solve the optimization algorithm. Iterates until a terminating condition
        is met. Terminating conditions are:
            * vanishing gradient
            * vanishing step size
            * max iterations surpassed
            * nans or infinities found in gradient or function value
        Each iteration includes the following:
            * check if any terminating condition has been met
            * take a step (varies by subclass)
            * update values associated with our new solution

        :return: a dictionary with the following keys and values
            hypothesis: the best hypothesis found for classifying the data
            objective: final objective value from training
            function_evaluations: how many times we evaluate our loss function
            gradient_norm: norm of the final gradient from training
            gradient_evaluations: how many times we evaluate our loss function's gradient
            iterations: how many iterations were completed
            status: what caused training to terminate (stalled, max iterations, optimal, numerical error)
            training_accuracy: how accurate is our final hypothesis
        """
        if self.track_errors:
            self.check_accuracy()

        while True:

            # check termination
            if self.norm_g <= self.tolerance:
                self.status = 'optimal'
                break
            elif self.iteration >= self.max_iterations:
                self.status = 'max iterations'
                break

            # step
            self.calculate_step()
            self.v_prev = self.v
            self.norm_vprev = self.norm_v
            self.v = self.v + self.p

            # update values associated with this iteration
            self.iteration += 1
            self.f = self.training_loss_function.f(self.v)
            self.g = self.evaluate_gradient()
            self.norm_v = np.linalg.norm(self.v)
            self.norm_g = np.linalg.norm(self.g)

            # check for NaN's and Infinities
            if not (np.isfinite(self.f).all() and np.isfinite(self.g).all()):
                self.status = 'numerical error'
                break

            # check if we're making reasonable progress
            if self.norm_p / (1 + self.norm_vprev) < sys.float_info.epsilon:
                self.status = 'stalled'
                break

            # status update
            if self.iteration % 20 == 0:
                if self.track_errors:
                    self.check_accuracy()
            if self.iteration % 100 == 0:
                print(f'iteration {self.iteration} norm_g: {self.norm_g}')

        # record training accuracy
        if self.iteration % 20 != 0 or not self.track_errors:
            self.check_accuracy()

        rtn = {
            'hypothesis': self.v,
            'objective': self.f,
            'function_evaluations': self.training_loss_function.f_calls,
            'gradient_norm': self.norm_g,
            'gradient_evaluations': self.training_loss_function.g_calls,
            'iterations': self.iteration,
            'status': self.status,
            'training error': self.training_error,
            'testing error': self.testing_error
        }

        return rtn

    def calculate_step(self):
        raise Exception('Please use a child class when running solve.')

    def evaluate_gradient(self):
        return self.training_loss_function.g(self.v)

    def check_accuracy(self):
        self.training_error[self.iteration] = 1 - self.training_loss_function.zero_one_loss(self.v)
        self.testing_error[self.iteration] = 1 - self.testing_loss_function.zero_one_loss(self.v)

    def _backtracking_armijo_line_search(self) -> float:
        """ conduct a backtracking armijo line-search to determine what proportion of
        the current step to take.

        :return: What proportion of the current step to take
        """
        a = self.a_init
        g_transpose_p = np.sum(self.g*self.p)
        while self.training_loss_function.f(self.v + a*self.p) > self.f + self.c * a * g_transpose_p:
            a *= self.eta
        return a


class CoordinateDescent(Algorithm):

    def __init__(self, block_size=20, *args, **kwargs):
        self.block_size = block_size
        self.coordinate_X_g = None
        self.u = None

        super().__init__(*args, **kwargs)

    def evaluate_gradient(self):
        """ Evaluate the gradient in coordinate descent.

        Calculates gradient for <self.block_size> random indices then lifts them
        back into the full space of the problem when returned.

        Caution: b/c b from logistic regression doesn't have coefficient in last
        dimension of X (it is added in the logistic regression instance), its never
        going to be selected for coordinate descent. If we did, we would have to
        assume all loss functions have a single axis term. This is why you just
        add it to the data set to begin with.

        :return:
        """
        # precompute Xv if we haven't already
        if self.u is None:
            X = np.append(self.training_loss_function.X,
                          np.ones((self.training_loss_function.m, 1)), axis=1)
            self.u = X @ self.v  # check these dimensions are right

        # random indices of our sample - subset of features on which to find gradient
        indices = np.random.choice(self.training_loss_function.n, size=self.block_size,
                                   replace=False)

        # calculate gradient for subset of indices
        coordinate_g = self.training_loss_function.g(self.v, col_indices=indices, u=self.u)

        # linear transformation lifting our coordinate gradient back into the full space
        U = np.zeros((self.v.shape[0], self.block_size))
        U[indices, range(self.block_size)] = np.ones(self.block_size)

        # transformation we'll use to update self.u, the precomputed Xv
        self.coordinate_X_g = self.training_loss_function.X[:, indices] @ coordinate_g

        return U @ coordinate_g  # lift gradient back into full dimension

    def calculate_step(self):
        """ Calculates the direction of the step and determines how far in that
        direction to step for coordinate descent with the initialized step length
        search method.

        :return:
        """
        # calculate step
        self.p = -self.g
        self.a = self._backtracking_armijo_line_search()
        self.p *= self.a
        self.norm_p = self.a * self.norm_g

        # update precomputation of Xv
        self.u = self.u - self.a * self.coordinate_X_g


class GradientDescent(Algorithm):

    def calculate_step(self):
        """ Calculates the direction of the step and determines how far in that
        direction to step for gradient descent with the initialized step length
        search method.

        :return:
        """
        # calculate step
        self.p = -self.g
        self.a = self._backtracking_armijo_line_search()
        self.p *= self.a
        self.norm_p = self.a * self.norm_g


class AcceleratedGradientDescent(Algorithm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 1
        self.u = None

    def calculate_step(self):
        """Calculates the direction of the step and determines how far in that
        direction to step for accelerated gradient descent with the initialized
        step length search method.

        :return:
        """
        # calculate parameters for accelerated gradient
        t_prev = self.t
        self.t = (1 + (1 + 4*self.t**2)**.5)/2
        beta = (t_prev - 1)/self.t

        # calculate step
        self.u = self.v + beta*(self.v - self.v_prev)
        self.p = -self.training_loss_function.g(self.u)
        self.a = self._backtracking_armijo_line_search()
        self.p *= self.a
        self.norm_p = np.linalg.norm(self.p)

    def _backtracking_armijo_line_search(self) -> float:
        """ conduct a backtracking armijo line-search to determine what proportion of
        the current step to take in the case we are running accelerated gradient descent

        :return: What proportion of the current step to take
        """
        a = self.a_init
        g_transpose_p = -np.sum(self.p**2)
        f = self.training_loss_function.f(self.u)
        while self.training_loss_function.f(self.u + a*self.p) > f + self.c * a * g_transpose_p:
            a *= self.eta
        return a


class StochasticGradientDescent(Algorithm):
    def __init__(self, training_loss_function: Logistic, b: int, *args, **kwargs):
        """

        :param training_loss_function: Which loss function to use
        :param b: number of samples for which we calculate gradients in each iteration
        :param args:
        :param kwargs:
        """
        assert isinstance(training_loss_function, Logistic), 'Loss function must be logistic'
        assert isinstance(b, int) and 0 < b <= training_loss_function.n, \
            'b must be a positive integer at most the number of samples'
        self.b = b
        v0 = np.zeros((training_loss_function.n + 1, 1))  # n for dim of each sample, 1 for y intercept
        super().__init__(training_loss_function, v0, *args, **kwargs)

    def evaluate_gradient(self):
        # random indices of our sample
        indices = np.random.choice(self.training_loss_function.m, size=self.b, replace=False)
        return self.training_loss_function.g(self.v, indices)

    def calculate_step(self):
        """ Calculates the direction of the step and determines how far in that
        direction to step for gradient descent with the initialized step length
        search method.

        :return:
        """
        # calculate step
        self.p = -self.g
        self.a = self.a_init/(self.iteration + 1)
        self.p *= self.a
        self.norm_p = self.a * self.norm_g


class StochasticHeavyBall(StochasticGradientDescent):
    def __init__(self, beta: float, *args, **kwargs):
        """

        :param beta: how much weight to give to the momentum term
        :param args:
        :param kwargs:
        """
        assert 0 <= beta < 1
        self.beta = beta
        super().__init__(*args, **kwargs)

    def calculate_step(self):
        """ Calculates the step to take as a combination of decreasing stochastic
        gradient and momentum.

        :return:
        """
        # calculate step
        self.a = self.a_init / (self.iteration + 1)
        self.p = -self.a*self.g + self.beta*(self.v - self.v_prev)
        self.norm_p = np.linalg.norm(self.p)


class ADAM(StochasticGradientDescent):
    def __init__(self, beta1: float, beta2: float, epsilon: float, *args, **kwargs):
        """

        :param beta1: how much weight to give to the previous 1st moment approximation
        :param beta2: how much weight to give to the previous 2nd moment approximation
        :param epsilon:
        :param args:
        :param kwargs:
        """
        assert 0 <= beta1 < 1
        self.beta1 = beta1
        assert 0 <= beta2 < 1
        self.beta2 = beta2
        assert 0 < epsilon < 1
        self.epsilon = epsilon

        super().__init__(*args, **kwargs)

        # current moment approximations
        self.m1 = np.zeros([self.training_loss_function.n + 1, 1])
        self.m2 = np.zeros([self.training_loss_function.n + 1, 1])
        self.m1_hat = None
        self.m2_hat = None

    def calculate_step(self):
        """ Calculates the step to take as a combination of decreasing stochastic
        gradient and momentum.

        :return:
        """
        # update moments
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.g
        self.m1_hat = self.m1 / (1 - self.beta1)
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * (self.g @ self.g.T).diagonal().reshape(-1, 1)
        self.m2_hat = self.m2 / (1 - self.beta2)

        # calculate step
        self.a = self.a_init / (self.iteration + 1)
        self.p = -self.a*self.m1_hat/ \
            (self.m2_hat**.5 + self.epsilon*np.ones((self.training_loss_function.n + 1, 1)))
        self.norm_p = np.linalg.norm(self.p)


