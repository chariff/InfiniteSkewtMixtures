import pytest
import mock

from InfiniteSkewtMixtures.bayesian_skew_t_mixture import BayesianSkewtMixture
from InfiniteSkewtMixtures.multivariate_skew_t import multivariate_skew_t as rv
import numpy as np

# skewt mixture number of observations and dimensions
n_samples = 1000
n_components = 4
dim = 2

# skewt mixture parameters
weights = np.array([.15, .05, .5, .3])
# locations
locs = np.array([[-2, 1.5], [1, 1], [1.5, -2], [-2, -2]])
shapes = np.array([[-.6, .6], [.8, .8], [2, -2], [-.8, -.8]])
scales = np.tile(np.identity(dim), n_components)
scales = scales.T.reshape(n_components, dim, dim) / 5
# degrees of freedom
dfs = np.array([50, 50, 50, 15])


# function to generate a skew t mixture
def gen_skewt_mixture(n_samples, n_components, dim, weights,
                      locs, shapes, scales, dfs, random_state=None):
    sizes = np.array(n_samples * weights, dtype=int)
    labels = np.repeat(np.arange(n_components), sizes)
    sample = np.zeros(shape=(dim, n_samples), dtype=float)
    prev_size = 0
    for k in range(n_components):
        size_k = sizes[k]
        component = rv.rvs(size_k, locs[k], shapes[k],
                           scales[k], dfs[k], random_state)
        sample[:, prev_size: prev_size + size_k] = component
        prev_size += size_k
    return sample.T, labels


def test_bayesian_skewt_mixture():
    # Generate a skew t mixture
    data_samples, labels = gen_skewt_mixture(n_samples, n_components, dim, weights,
                                             locs, shapes, scales, dfs,
                                             random_state=2020)

    max_iter = 5
    burn_in = 1

    # random init
    cls = BayesianSkewtMixture(max_iter=5, burn_in=1,
                               verbose=2, verbose_interval=500,
                               random_state=2020, init_params='random')

    p = cls.fit(data_samples)
    map_partition = p.map_partition
    mcmc_partitions = p.partitions

    assert map_partition.shape == (n_samples, )
    assert len(mcmc_partitions) == max_iter - burn_in

    # kmeans init
    cls = BayesianSkewtMixture(max_iter=5, burn_in=1,
                               verbose=2, verbose_interval=500,
                               random_state=2020, init_params='kmeans')

    p = cls.fit(data_samples)
    map_partition = p.map_partition
    mcmc_partitions = p.partitions

    assert map_partition.shape == (n_samples, )
    assert len(mcmc_partitions) == max_iter - burn_in


if __name__ == '__main__':
    pytest.main([__file__])