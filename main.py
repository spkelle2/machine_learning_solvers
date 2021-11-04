import numpy as np
import pandas as pd
import scipy.io

from algorithms import GradientDescent, AcceleratedGradientDescent, \
    StochasticGradientDescent, StochasticHeavyBall, ADAM, CoordinateDescent
from loss_functions import Logistic


def main_hw2():
    # load data
    mat = scipy.io.loadmat('data/mnist.mat')
    y = mat['data'][0][0][0]
    X = mat['data'][0][0][1]
    y_train = y[:80000]
    y_test = y[80000:]
    X_train = X[:80000]
    X_test = X[80000:]

    # run gradient descent
    print('--------- Using Gradient Descent ---------')
    gd_train_loss_function = Logistic(X=X_train, y=y_train)
    gd_test_loss_function = Logistic(X=X_test, y=y_test)
    v0 = np.zeros((gd_train_loss_function.n + 1, 1))
    gd = GradientDescent(training_loss_function=gd_train_loss_function, v0=v0)
    gd_rtn = gd.solve()

    print('\n \n --- Gradient Descent Training Results ---')
    print(gd_rtn)

    print('\n \n --- Gradient Descent Testing Results ---')
    gd_test_loss_function.zero_one_loss(gd.v)

    # run accelerated gradient descent
    print('\n --------- Using Accelerated Gradient Descent ---------')
    agd_train_loss_function = Logistic(X=X_train, y=y_train)
    agd_test_loss_function = Logistic(X=X_test, y=y_test)
    agd = AcceleratedGradientDescent(loss_function=agd_train_loss_function, v0=v0)
    agd_rtn = agd.solve()

    print('\n \n --- Accelerated Gradient Descent Training Results ---')
    print(agd_rtn)

    print('\n \n --- Accelerated Gradient Descent Testing Results ---')
    agd_test_loss_function.zero_one_loss(agd.v)


def run_stochastic_gradient_descent(X_train, X_test, y_train, y_test):
    # run stochastic gradient descent
    print('----- Using Stochastic Gradient Descent -----')
    sgd_train_loss_function = Logistic(X=X_train, y=y_train)
    sgd_test_loss_function = Logistic(X=X_test, y=y_test)
    data = {}
    for b in [32, 64, 128]:
        for a_init in [.01, .1, 1]:
            print(f'\n b: {b}, a: {a_init}')
            sgd = StochasticGradientDescent(
                training_loss_function=sgd_train_loss_function, b=b, a_init=a_init,
                max_iterations=10*sgd_test_loss_function.n,
                testing_loss_function=sgd_test_loss_function, track_errors=True
            )
            sgd_rtn = sgd.solve()
            # append this test to our file
            train_df = pd.DataFrame.from_dict(sgd_rtn['training error'], orient='index')
            test_df = pd.DataFrame.from_dict(sgd_rtn['testing error'], orient='index')
            for name, df in {'train': train_df, 'test': test_df}.items():
                df.reset_index(inplace=True)
                df.columns = ['iteration', 'error']
                df.insert(0, 'b', [b] * len(sgd_rtn[f'{name}ing error']))
                df.insert(0, 'alpha', [a_init] * len(sgd_rtn[f'{name}ing error']))

                with open(f'stochastic_gradient_descent_{name}_comparison.csv', 'a') as f:
                    df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
            data[b, a_init] = sgd_rtn
    return data


def run_stochastic_heavy_ball(X_train, X_test, y_train, y_test):
    # run stochastic heavy ball
    print('\n\n----- Using Stochastic Heavy Ball Method -----')
    train_loss_function = Logistic(X=X_train, y=y_train)
    test_loss_function = Logistic(X=X_test, y=y_test)
    data = {}
    for b in [32, 64, 128]:
        for a_init in [.01, .1, 1]:
            for beta in [.1, .3, .5]:
                print(f'\n b: {b}, a: {a_init}, beta: {beta}')
                algorithm = StochasticHeavyBall(
                    training_loss_function=train_loss_function, b=b, a_init=a_init,
                    max_iterations=10*test_loss_function.n,
                    testing_loss_function=test_loss_function, track_errors=True,
                    beta=beta
                )
                algorithm_rtn = algorithm.solve()
                # append this test to our file
                train_df = pd.DataFrame.from_dict(algorithm_rtn['training error'], orient='index')
                test_df = pd.DataFrame.from_dict(algorithm_rtn['testing error'], orient='index')
                for name, df in {'train': train_df, 'test': test_df}.items():
                    df.reset_index(inplace=True)
                    df.columns = ['iteration', 'error']
                    df.insert(0, 'b', [b] * len(algorithm_rtn[f'{name}ing error']))
                    df.insert(0, 'alpha', [a_init] * len(algorithm_rtn[f'{name}ing error']))
                    df.insert(0, 'beta', [beta] * len(algorithm_rtn[f'{name}ing error']))

                    with open(f'stochastic_heavy_ball_{name}_comparison.csv', 'a') as f:
                        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
                data[b, a_init, beta] = algorithm_rtn
    return data


def run_adam(X_train, X_test, y_train, y_test):
    # run adam
    print('\n\n----- Using ADAM -----')
    train_loss_function = Logistic(X=X_train, y=y_train)
    test_loss_function = Logistic(X=X_test, y=y_test)
    data = {}
    for b in [32, 64, 128]:
        for a_init in [.01, .1, 1]:
            for beta1 in [.1, .5, .9]:
                for beta2 in [.1, .5, .9]:
                    print(f'\n b: {b}, a: {a_init}, beta1: {beta1}, beta2: {beta2}')
                    algorithm = ADAM(
                        training_loss_function=train_loss_function, b=b, a_init=a_init,
                        max_iterations=10*test_loss_function.n,
                        testing_loss_function=test_loss_function, track_errors=True,
                        beta1=beta1, beta2=beta2, epsilon=.0001
                    )
                    algorithm_rtn = algorithm.solve()
                    # append this test to our file
                    train_df = pd.DataFrame.from_dict(algorithm_rtn['training error'], orient='index')
                    test_df = pd.DataFrame.from_dict(algorithm_rtn['testing error'], orient='index')
                    for name, df in {'train': train_df, 'test': test_df}.items():
                        df.reset_index(inplace=True)
                        df.columns = ['iteration', 'error']
                        df.insert(0, 'b', [b] * len(algorithm_rtn[f'{name}ing error']))
                        df.insert(0, 'alpha', [a_init] * len(algorithm_rtn[f'{name}ing error']))
                        df.insert(0, 'beta1', [beta1] * len(algorithm_rtn[f'{name}ing error']))
                        df.insert(0, 'beta2', [beta2] * len(algorithm_rtn[f'{name}ing error']))

                        with open(f'adam_{name}_comparison.csv', 'a') as f:
                            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
                    data[b, a_init, beta1, beta2] = algorithm_rtn
    return data


def main_hw3():
    # load data
    mat = scipy.io.loadmat('data/mnist.mat')
    y = mat['data'][0][0][0]
    X = mat['data'][0][0][1]
    y_train = y[:80000]
    y_test = y[80000:]
    X_train = X[:80000]
    X_test = X[80000:]
    data = {}

    data['sgd'] = run_stochastic_gradient_descent(X_train, X_test, y_train, y_test)
    data['shb'] = run_stochastic_heavy_ball(X_train, X_test, y_train, y_test)
    data['adam'] = run_adam(X_train, X_test, y_train, y_test)
    print()


def main_hw4():
    # load data
    mat = scipy.io.loadmat('data/mnist.mat')
    y = mat['data'][0][0][0]
    X = mat['data'][0][0][1]
    y_train = y[:80000]
    y_test = y[80000:]
    X_train = X[:80000]
    X_test = X[80000:]

    # run coordinate descent
    print('----- Using Coordinate Descent -----')
    cd_train_loss_function = Logistic(X=X_train, y=y_train)
    cd_test_loss_function = Logistic(X=X_test, y=y_test)
    cd = CoordinateDescent(training_loss_function=cd_train_loss_function,
                           testing_loss_function=cd_test_loss_function, track_errors=True
    )
    cd_rtn = cd.solve()
    # append this test to our file
    train_df = pd.DataFrame.from_dict(cd_rtn['training error'], orient='index')
    test_df = pd.DataFrame.from_dict(cd_rtn['testing error'], orient='index')
    for name, df in {'train': train_df, 'test': test_df}.items():
        df.reset_index(inplace=True)
        df.columns = ['iteration', 'error']
        with open(f'coordinate_descent_{name}_comparison.csv', 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    return None


if __name__ == '__main__':
    main_hw4()
