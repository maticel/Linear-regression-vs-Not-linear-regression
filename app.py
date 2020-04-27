from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(input_data):
    data = pd.read_csv(input_data, header=None, sep=' ').drop(2, axis=1)
    train_data, test_data = train_test_split(data, random_state=33, train_size=0.75)
    return train_data.to_numpy(), test_data.to_numpy()


def linreg(data):
    n = data.shape[0]
    x = np.sum(data[:, 0])
    y = np.sum(data[:, 1])
    xy = np.sum(data[:, 0] * data[:, 1])
    x_pow2 = np.sum(data[:, 0] ** 2)

    w1 = ((n * xy) - (x * y)) / ((n * x_pow2) - (x ** 2))
    w0 = ((y * x_pow2) - (x * xy)) / ((n * x_pow2) - (x ** 2))
    return w1, w0


def notlinreg(data):  # square regression
    n = data.shape[0]
    x = np.sum(data[:, 0])
    y = np.sum(data[:, 1])
    xy = np.sum(data[:, 0] * data[:, 1])
    x_pow2y = np.sum((data[:, 0] ** 2) * data[:, 1])
    x_pow2 = np.sum(data[:, 0] ** 2)
    x_pow3 = np.sum(data[:, 0] ** 3)
    x_pow4 = np.sum(data[:, 0] ** 4)

    inverse_matrix = np.linalg.inv(np.array([
        [x_pow4, x_pow3, x_pow2],
        [x_pow3, x_pow2, x],
        [x_pow2, x, float(n)]]))

    matrix = np.array([
        [x_pow2y],
        [xy],
        [y]])

    w = inverse_matrix.dot(matrix)
    return np.concatenate(w)


def mean_square_error(w, data):
    x = data[:, 0]
    y = np.zeros(len(data))
    for i in range(len(w)):
        y += w[i] * (x ** (len(w)-1-i))
    error = np.sum((data[:, 1] - y) ** 2)/data.shape[0]
    return error


def crtplot(w_lin, w_notlin, data):
    X = data[:, 0]
    Y = data[:, 1]
    x = np.linspace(np.min(X), np.max(X), 10)

    y_lin = w_lin[0] * x + w_lin[1]
    y_notlin = w_notlin[0] * (x ** 2) + w_notlin[1] * x + w_notlin[2]

    mse_lin = mean_square_error(w_lin, data)
    mse_notlin = mean_square_error(w_notlin, data)

    plt.plot(x, y_lin, color='red',
             label='Linear regression line MSE={0:.0%}'.format(mse_lin))

    plt.plot(x, y_notlin, color='magenta',
             label='Not linear regression line MSE={0:.0%}'.format(mse_notlin))

    plt.scatter(X, Y, color='green', label='Point')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


def main(input_data):
    train_data, test_data = prepare_data(input_data)

    # linear regression
    w_lin = linreg(train_data)

    # not linear regression
    w_notlin = notlinreg(train_data)

    crtplot(w_lin, w_notlin, test_data)


def parse_arguments():
    parser = ArgumentParser(description='description')
    parser.add_argument('-i', '--input_data', type=str, required=True, help='help')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.input_data)
