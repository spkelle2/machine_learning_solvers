import matplotlib.pyplot as plt
import pandas as pd

def hw3_graphs():
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    file = 'stochastic_gradient_descent_train_comparison.csv'
    df_sgd_train = pd.read_csv(file)
    df_sgd_train = df_sgd_train.loc[(df_sgd_train['b'] == 128) & (df_sgd_train['alpha'] == .1)]  # alpha .1 b 128
    df_sgd_train = df_sgd_train[['iteration', 'error']]
    ax1.plot(df_sgd_train['iteration'], df_sgd_train['error'], label='SGD')

    file = 'stochastic_gradient_descent_test_comparison.csv'
    df_sgd_test = pd.read_csv(file)
    df_sgd_test = df_sgd_test.loc[(df_sgd_test['b'] == 128) & (df_sgd_test['alpha'] == .1)]
    df_sgd_test = df_sgd_test[['iteration', 'error']]
    ax2.plot(df_sgd_test['iteration'], df_sgd_test['error'], label='SGD')

    file = 'stochastic_heavy_ball_train_comparison.csv'
    df_shb_train = pd.read_csv(file)
    df_shb_train = df_shb_train.loc[(df_shb_train['b'] == 64) & (df_shb_train['alpha'] == .1) & (df_shb_train['beta'] == .3)]
    df_shb_train = df_shb_train[['iteration', 'error']]
    ax1.plot(df_shb_train['iteration'], df_shb_train['error'], label='SHB')

    file = 'stochastic_heavy_ball_test_comparison.csv'
    df_shb_test = pd.read_csv(file)
    df_shb_test = df_shb_test.loc[(df_shb_test['b'] == 64) & (df_shb_test['alpha'] == .1) & (df_shb_test['beta'] == .3)]
    df_shb_test = df_shb_test[['iteration', 'error']]
    ax2.plot(df_shb_test['iteration'], df_shb_test['error'], label='SHB')

    file = 'adam_train_comparison.csv'
    df_adam_train = pd.read_csv(file)
    df_adam_train = df_adam_train.loc[(df_adam_train['b'] == 32) & (df_adam_train['alpha'] == 1)
                                      & (df_adam_train['beta1'] == .9) & (df_adam_train['beta2'] == .1)]
    df_adam_train = df_adam_train[['iteration', 'error']]
    ax1.plot(df_adam_train['iteration'], df_adam_train['error'], label='ADAM')

    file = 'adam_test_comparison.csv'
    df_adam_test = pd.read_csv(file)
    df_adam_test = df_adam_test.loc[(df_adam_test['b'] == 32) & (df_adam_test['alpha'] == 1)
                                    & (df_adam_test['beta1'] == .9) & (df_adam_test['beta2'] == .1)]
    df_adam_test = df_adam_test[['iteration', 'error']]
    ax2.plot(df_adam_test['iteration'], df_adam_test['error'], label='ADAM')

    ax1.set_title("Training Comparison")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Error")
    ax1.legend()

    ax2.set_title("Testing Comparison")
    ax2.set_xlabel("Iterations")
    ax2.legend()

    fig2.show()


def hw4_graphs():
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    file = 'coordinate_descent_train_comparison.csv'
    df_cd_train = pd.read_csv(file)
    ax1.plot(df_cd_train['iteration'], df_cd_train['error'])

    file = 'coordinate_descent_test_comparison.csv'
    df_cd_test = pd.read_csv(file)
    ax2.plot(df_cd_test['iteration'], df_cd_test['error'])

    ax1.set_title("Coord. Descent Training Error")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Error")

    ax2.set_title("Coord. Descent Testing Error")
    ax2.set_xlabel("Iterations")

    fig2.show()

    print()


if __name__ == '__main__':
    hw4_graphs()
