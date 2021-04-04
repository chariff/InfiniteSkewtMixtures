

# Bayesian Infinite Skew-t Mixtures.


[![Build Status](https://travis-ci.org/chariff/InfiniteSkewtMixtures.svg?branch=master)](https://travis-ci.org/chariff/InfiniteSkewtMixtures)
[![Codecov](https://codecov.io/github/chariff/InfiniteSkewtMixtures/badge.svg?branch=master&service=github)](https://codecov.io/github/chariff/InfiniteSkewtMixtures?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)




Python implementation of a Bayesian nonparametric approach of skew-t distributions to perform model based clustering.   
The Dirichlet process prior on the mixing distribution allows for an inference of the number of classes directly 
from  the  data thus avoiding  model selection issues. Skew-t distributions provide robustness to outliers and  
non-elliptical shape of clusters.  

Installation.
============

### Installation

* From GitHub:

      pip install git+https://github.com/chariff/InfiniteSkewtMixtures.git

### Dependencies
GPro requires:
* Python (>= 3.5)
* NumPy+mkl (>= 1.18.5)
* SciPy+mkl (>= 1.4.1)
* sklearn (>= 0.23.2)


Brief guide to using Bayesian Infinite Skew-t Mixtures.
=========================

Checkout the package docstrings for more information.

## 1. Fitting an infinite skew-t mixture model and making predictions.

```python
from InfiniteSkewtMixtures.bayesian_skew_t_mixture import BayesianSkewtMixture
from InfiniteSkewtMixtures.multivariate_skew_t import multivariate_skew_t as rv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.style.use('seaborn')
```
Training data will be simulated from the function below.
A minimum of two values is required. The following example is 
a skew-t mixture of 4 components in 2D.
```python
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
        sample[:, prev_size : prev_size + size_k] = component
        prev_size += size_k
    return sample.T, labels
    
# Generate a skew t mixture
data_samples, labels = gen_skewt_mixture(n_samples, n_components, dim, weights,
                                         locs, shapes, scales, dfs,
                                         random_state=2020)

```
The simulated data, 'data_samples' is an array of shape (n_observations, n_features).
labels is an array of shape (n_observations, ) and can be used
to compare the estimated partition.

Select the number maximum number of mcmc (Monte Carlo Markov Chain) iterations.
Important note : The number of mcmc iterations must be carefully chosen. Theoretically, 
as the number of iterations increases, the mcmc sample should converge to the target distribution. 
The more iterations you can afford, the better. A variational algorithm would be faster but at the 
expense of not having a reliable convergence criteria. 
```python
# Maximum number of mcmc iterations 
max_iter = 2000
# Burn-in iterations
burn_in = 1500
```
Instantiate a BayesianSkewtMixture object with default parameters.
```python
cls = BayesianSkewtMixture(max_iter=max_iter, burn_in=burn_in,
                           verbose=2, verbose_interval=500,
                           random_state=2020, init_params='random')
```
Fit a Bayesian infinite skew-t mixture model. 
```python
p = cls.fit(data_samples)
```
Maximum a posteriori (MAP) partition.
```python
map_partition = p.map_partition
```
```python
# Scatter plot of the MAP partition
for label in set(map_partition):
    plt.scatter(data_samples[map_partition == label, 0],
                data_samples[map_partition == label, 1], 
                c=next(color_cycle), alpha=.5, s=20, 
                label='class ' + str(label))
plt.title('MAP partition')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(loc='upper right')
plt.show()
```
![MAP partition](https://github.com/chariff/BayesianInfiniteMixtures/raw/master/examples/MAP_partition_0.png)

Log posterior trace.
```python
plt.plot(p.logposteriors,  linewidth=.6, c='black')
plt.title('Log posterior trace')
plt.xlabel('mcmc iterations')
plt.ylabel('Log posterior evaluation')
plt.show()
```
![Log posterior trace](https://github.com/chariff/BayesianInfiniteMixtures/raw/master/examples/trace_0.png)

Predict new values.
```python
# predict data_samples for the sake of example
map_predicted_partition = p.map_predict(data_samples)
```

The MAP approach ignores the mcmc partitions sample and clustering uncertainty cannot be assessed.
In the following section we will show an example of how to use the sampled partitions to assess
the clustering uncertainty in a case with overlap between two distributions.
```python
# skewt mixture number of observations and dimensions
n_samples = 1000
n_components = 2
dim = 2

# skewt mixture parameters
weights = np.array([.6, .4])
# locations
locs = np.array([[1.5, -2], [2, 2]])
shapes = np.array([[-2, 2], [-.8, -.8]])
scales = np.tile(np.identity(dim), n_components)
scales = scales.T.reshape(n_components, dim, dim) / 5
# degrees of freedom
dfs = np.array([50, 50])

# Generate a skew t mixture
data_samples, labels = gen_skewt_mixture(n_samples, n_components, dim, weights,
                                         locs, shapes, scales, dfs,
                                         random_state=2020)

```
```python
# Maximum number of mcmc iterations 
max_iter = 2000
# Burn-in iterations
burn_in = 1500
cls = BayesianSkewtMixture(max_iter=max_iter, burn_in=burn_in,
                           verbose=2, verbose_interval=500,
                           random_state=2020, init_params='random')
```
Fit a Bayesian infinite skew-t mixture model. 
```python
p = cls.fit(data_samples)
```

```python
for label in set(p.map_partition):
    plt.scatter(data_samples[p.map_partition == label, 0],
                data_samples[p.map_partition == label, 1], 
                c=next(color_cycle), alpha=.5, s=20, 
                label='class ' + str(label))
plt.title('Case with overlap')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(loc='upper right')
plt.show()

```
MAP partition.
![MAP partition](https://github.com/chariff/BayesianInfiniteMixtures/raw/master/examples/MAP_partition_overlap_1.png)

The following function enables to calculate an average of the co-clustering
matrices from the explored partitions in the posterior mcmc draws
to obtain the posterior co-clustering probabilities.
It could be parallelised on the partitions but it would still have a quadratic 
computational cost. 
```python
from numba import jit

@jit(nopython=True)
def coclustering(partitions):
    n_mcmc_samples, n_samples = partitions.shape
    co_clustering = np.zeros(shape=(n_samples, n_samples))
    for partition in partitions:
        for i in range(n_samples):
            label_obs = partition[i]
            for j in range(n_samples):
                if partition[j] == label_obs:
                    co_clustering[i, j] += 1
    co_clustering /= n_mcmc_samples
    return co_clustering
```
```python
# explored partitions in the posterior mcmc draws
partitions = p.partitions
# compute the average co-clustering matrix
coclust = coclustering(partitions)
```
Heatmap of the posterior co-clustering probabilities.
```python
# Plot co-clustering matrix.
fig = plt.figure(figsize=(8, 8))
im = plt.matshow(coclust, aspect='auto', origin='lower', cmap=cm.OrRd)
plt.colorbar(im)
plt.title('Average co-clustering matrix')
plt.xlabel('observations')
plt.ylabel('observations')
plt.show()
```
![Co-clustering probabilities](https://github.com/chariff/BayesianInfiniteMixtures/raw/master/examples/avg_coclust.png)

To obtain a point estimate of the clustering, one
could minimizes a loss function of the co-clustering matrices from the 
explored partitions in the posterior mcmc draws and the 
posterior co-clustering probabilities.

### References:
* https://academic.oup.com/biostatistics/article/11/2/317/268224
* https://projecteuclid.org/euclid.aoas/1554861663
* https://projecteuclid.org/euclid.aos/1056562461
* https://www.tandfonline.com/doi/abs/10.1080/03610910601096262
* https://link.springer.com/article/10.1007/s11222-009-9150-y
* https://www.stat.berkeley.edu/~pitman/621.pdf
* https://www.jstor.org/stable/24305538?seq=1
* http://www2.stat.duke.edu/~mw/MWextrapubs/West1992alphaDP.pdf


    -- Chariff Alkhassim