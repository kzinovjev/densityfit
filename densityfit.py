#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit


def expand_sigma(sigma1d):
    d = int(np.sqrt(len(sigma1d) * 2))
    sigma = np.zeros((d, d))
    idx = np.triu_indices(d)
    sigma[idx] = sigma1d
    sigma.T[idx] = sigma1d
    return sigma


def quad_form(x, a):
    return np.einsum('ij,jk,ik->i', x, a, x)


def gaussian(x, a, mu, sigma):
    det_sigma = np.linalg.det(sigma)
    if det_sigma < 0:
        return np.full(len(x), np.inf)
    inv_sigma = np.linalg.inv(sigma)
    prefactor = a / np.sqrt(det_sigma * (np.pi * 2) ** len(mu))
    return prefactor * np.exp(-quad_form(x - mu, inv_sigma) / 2)


def multi_gaussian_factory(d):
    params_size = (d + 2) * (d + 1) // 2

    def multi_gaussian(x, *params):
        params_chunks = [params[i:i + params_size]
                         for i in range(0, len(params), params_size)]
        return np.array([gaussian(x, p[0], p[1:d + 1], expand_sigma(p[d + 1:]))
                         for p in params_chunks]).sum(axis=0)

    return multi_gaussian


def centers_to_bins(edges, centers):
    return np.array([np.searchsorted(edges[i], centers.T[i])
                     for i in range(edges.shape[0])]).T - 1


def get_single_guess(hist, center_bin, center, sigma_diag):
    sigma = np.diag(sigma_diag)[np.triu_indices(len(sigma_diag))]
    a = hist[tuple(center_bin)]
    if a == 0:
        a = np.min(hist[hist > 0])
    return np.array([a, *center, *sigma])


def get_guess(hist, edges, centers):
    sigma_diag = edges[:, 1] - edges[:, 0]
    center_bins = centers_to_bins(edges, centers)
    return np.array([
        get_single_guess(hist, center_bin, center, sigma_diag)
        for center_bin, center
        in zip(center_bins, centers)
    ]).flatten()


def bin_to_x(bin_mids, b):
    return [bin_mids[i_bin] for i_bin in enumerate(b)]


def fit_histogram(hist, edges, centers, thr=0):
    edges = np.array(edges)
    bin_mids = (edges[:, :-1] + edges[:, 1:]) / 2

    vals = hist[hist > thr]
    bins = np.argwhere(hist > thr)
    x = np.array([[bin_mids[i_bin] for i_bin in enumerate(b)] for b in bins])

    guess = get_guess(hist, edges, np.array(centers))

    n_dims = len(edges)
    n_centers = len(centers)
    n_params = len(guess)

    l_bounds = np.full(n_params, -np.inf)
    l_bounds[np.arange(0, n_params, n_params // n_centers)] = 0

    f = multi_gaussian_factory(n_dims)
    final_params, cov = curve_fit(f=f,
                                  xdata=x,
                                  ydata=vals,
                                  p0=guess,
                                  bounds=(l_bounds, np.inf))

    fit = np.zeros(hist.shape)
    for index in np.ndindex(*hist.shape):
        fit[index] = f(np.array([bin_to_x(bin_mids, index)]), *final_params)

    return final_params, cov, bin_mids, fit


def fit_sample(data, centers, bins=10, thr=0):
    hist, edges = np.histogramdd(data, bins, density=True)
    return fit_histogram(hist, edges, centers, thr)


def arg_to_data(arg, d):
    arr = np.array([float(x) for x in arg.split(',')])
    return arr.reshape((len(arr) // d, d))


def main():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Fits a set of multivariate gaussians to a data sample"
    )
    parser.add_argument("data_file",
                        help="Data file with each row being a single data point")
    parser.add_argument("centers",
                        help="Initial guess consisting of comma separated "
                             "coordinates for centers of the gaussians")
    parser.add_argument("bins", type=int, help="Number of bins", nargs='?')
    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        raw_data = [[float(x) for x in line.split()] for line in f]
    data = tuple(np.array(raw_data).T)

    d = len(data)
    centers = arg_to_data(args.centers, d)

    n_bins = int(args.bins) if args.bins else 30

    final_params, cov, bin_mids, fit = fit_sample(data, centers, n_bins)

    n_params = len(final_params) // len(centers)
    params = final_params.reshape((len(centers), n_params))
    print(params)

    if d > 2:
        return

    fig, ax = plt.subplots()
    if d == 1:
        ax.hist(data[0], bins=n_bins, density=True)
        ax.plot(bin_mids[0], fit)
        ax.scatter(params[:, 1], params[:, 0], marker='o', c="red", zorder=10)
    if d == 2:
        ax.contour(bin_mids[0], bin_mids[1], fit.T, levels=20)
        ax.scatter(data[0], data[1], marker=',', s=1,
                   linewidths=0, alpha=0.25, c="black")
        ax.scatter(params[:, 1], params[:, 2], marker='o', c="red", zorder=10)
    plt.show()


if __name__ == '__main__':
    main()
