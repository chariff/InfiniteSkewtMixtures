# Author: Chariff Alkhassim <chariff.alkhassim@gmail.com>
# License: MIT

import sys

import numpy as np

from scipy.stats import (beta, gamma, multivariate_normal,
                         uniform, invwishart, wishart)
from scipy.special import (loggamma, polygamma, ndtr, ndtri)
from scipy.linalg import cholesky, eigvalsh

from multivariate_skew_t import multivariate_skew_t as rv
from multivariate_skew_t import PSD
from gibbs_sampling_base import GibbsSamplingMixture

from collections import Counter

from sklearn.utils import check_array


_C_DF_MH = 2
_MACHINE_FLOAT_MIN = sys.float_info.min
_MACHINE_FLOAT_EPS = np.finfo('float').eps
_RANDOM_STATE_UPPER = 1e5


def _check_scale_matrix(scale):
    """Check if a scale matrix is symmetric and positive-definite."""

    if not (np.allclose(scale, scale.T) and
            np.all(eigvalsh(scale) > 0.)):
        raise ValueError("scale should be symmetric, positive-definite")


def _check_positive_scalar(x, name):
    """Check if a value x is a positive scalar."""

    if not x > 0 and np.isscalar(x):
        raise ValueError("The parameter '%s' should be a positive scalar, "
                         "but got %s" % (name, x))


def _clusters_sizes(labels, max_components):
    """Compute the number of components and the size of
       samples in each component given an array of labels.

    Parameters
    ----------
    labels : array of shape (n_observations, )
        The labels of each observation.

    max_components : int
        Maximum number of components.

    Returns
    -------
    components_sizes : array of shape (max_components, )
        The size of samples in each component.

    n_components : int
        The number of components.

    components_labels : list, (n_components)
        The labels of each component.
    """

    components_sizes = np.zeros(max_components, dtype=int)
    table_labels = Counter(labels)
    n_components = len(table_labels)
    components_labels = list(table_labels.keys())
    for key, value in table_labels.items():
        components_sizes[key] = value

    return components_sizes, n_components, components_labels


def log_jeffrey_prior(x):
    """ Non-informative prior distribution for a parameter space

    Parameters
    ----------
    x : positive scalar

    Returns
    -------
    Logarithm transformation of the Jeffrey prior : float
    """

    p1 = polygamma(n=1, x=x / 2)
    p2 = polygamma(n=1, x=(x + 1) / 2) + 2 * (x + 3) / (x * (x + 1) ** 2)
    if p1 <= p2:
        return -1 * float('inf')
    return .5 * (np.log(x) - np.log(x + 3) + np.log(p1 - p2))


def _rtrunc_norm(mean, sd, lower, upper, random_state=None):
    """Sample from a truncated normal distribution

    Parameters
    ----------
    mean : float or array of shape (n_observations, )
        Means of the distribution.

    sd : float or array of shape (n_observations, )
        Standard deviations of the distribution.

    lower : float or array of shape (n_observations, )
        Lower bounds of the distribution.

    upper : float or array of shape (n_observations, )
        Upper bounds of the distribution.

    Note
    ----
    Arrays passed must all be of the same length. Computes samples
    using the Phi, the normal CDF, and Phi^{-1} using a standard
    algorithm:
    draw u ~ uniform(|Phi((l - m) / sd), |Phi((u - m) / sd))
    return m + sd * Phi^{-1}(u)

    Returns
    -------
    samples : float or array of shape (n_observations, )
    """

    u_lower = ndtr((lower - mean) / sd)
    u_upper = ndtr((upper - mean) / sd)
    draws = uniform.rvs(size=len(u_lower), random_state=random_state)
    u = (u_upper - u_lower) * draws + u_lower
    # prevent ndtriu from returning inf (integral over R)
    u[np.isclose(a=u, b=1, rtol=0, atol=1e-18, equal_nan=False)] = \
        1 - _MACHINE_FLOAT_EPS
    return mean + sd * ndtri(u)


def _structured_niw_posterior_params(X, scales_tn, loc_prior, shape_prior,
                                     scale_prior, w_loc_shape_prior,
                                     degrees_of_freedom_prior, trunc_normal,
                                     hierarchical_scale_prior, scale_inv,
                                     scale_prior_inv, n_components, n_features,
                                     allow_singular, random_state):
    """Structured Normal-inverse-Wishart posterior parameters estimation.
       Conjugate prior of a skew-t distribution for all its parameters except for the
       degree of freedom.

    Parameters
    ----------

    X : array of shape (n_observations, n_features)
        Observations modelled by a skew-t distribution.

    scales_tn : array of shape (n_observations, )
        Scales of the truncated normal regressors.

    loc_prior :  array of shape (n_features, )
        Location hyper parameter.

    shape_prior :  array of shape (n_features, )
        Shape hyper parameter.

    scale_prior : array of shape (n_features, n_features)
        Scale hyper parameter.

    w_loc_shape_prior : array of shape (2, 2)
        Diagonal matrix. The upper element controls the prior information
        in loc_prior. The lower element controls the prior information in
        shape_prior.

    degrees_of_freedom_prior : float
        Degree of freedom hyper parameter. Must be superior to n_features - 1.

    trunc_normal : array of shape (n_observations, )
        Regressors sampled from a truncated normal distribution.

    hierarchical_scale_prior : boolean
        Put a hierarchical Wishart distribution prior on the scale prior.

    scale_inv : array of shape (n_features, n_features)
        hyper parameter of hierarchical hyper prior Whishart on the scale prior.

    scale_prior_inv : array of shape (n_features, n_features)
        hyper parameter of hierarchical hyper prior Whishart on the scale prior.
        It is the sum of the skew-t mixture components scales.

    n_components : int
        Number of components of the skew-t mixture.

    n_features : int
        Number of dimensions of the skew-t mixture.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.

    Returns
    -------

    loc : array of shape (n_features, )

    shape : array of shape (n_features, )

    scale : array of shape (n_features, n_features)

    w_loc_shape : array of shape (2, 2)

    degree_of_freedom : float

    References
    ----------
    .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

    """
    n = len(X)

    # posterior degrees of freedom
    degrees_of_freedom = degrees_of_freedom_prior + n / 2

    # rescaled observations
    scales_tn_sqrt = np.sqrt(scales_tn)
    Xr = np.stack((scales_tn_sqrt, scales_tn_sqrt * trunc_normal), axis=-1)
    rescaled_obs = X * scales_tn_sqrt[:, np.newaxis]

    # posterior location, posterior shape, posterior control over them 'w_loc_shape'.
    w_loc_shape_priors_d = w_loc_shape_prior.diagonal()
    w_loc_shape_priors_inv = np.diag(1 / w_loc_shape_priors_d)
    loc_shape_part1 = np.transpose(rescaled_obs) @ Xr
    loc_shape_part2 = np.stack((loc_prior, shape_prior),
                               axis=-1) / w_loc_shape_priors_d
    psd_w_loc_shape = PSD(np.transpose(Xr) @ Xr + w_loc_shape_priors_inv,
                          allow_singular=allow_singular)
    w_loc_shape = psd_w_loc_shape.U @ np.transpose(psd_w_loc_shape.U)
    loc_shape = (loc_shape_part1 + loc_shape_part2) @ w_loc_shape
    loc, shape = loc_shape[:, 0], loc_shape[:, 1]

    # posterior scale
    epsilon = X - loc - trunc_normal[:, np.newaxis] * shape[np.newaxis, :]
    epsilon_c = epsilon[:, :, np.newaxis] @ epsilon[:, np.newaxis]
    scaled_eps = (epsilon_c * scales_tn.reshape(n, 1, 1)).sum(axis=0)
    delta_loc_loc_prior = (loc - loc_prior)[:, np.newaxis]
    delta_shape_shape_prior = (shape - shape_prior)[:, np.newaxis]
    w_loc_prior, w_shape_prior = w_loc_shape_priors_d
    scale_loc = \
        delta_loc_loc_prior @ np.transpose(delta_loc_loc_prior) / w_loc_prior
    scale_shape = \
        delta_shape_shape_prior @ np.transpose(delta_shape_shape_prior) / \
        w_shape_prior

    # hierarchical Wishart distribution prior on the scale prior.
    if hierarchical_scale_prior:
        g0 = .5 + (n_features - 1) / 2
        gn = g0 + n_components * degrees_of_freedom_prior
        if n_features > 1:
            scale_in_hyper = np.linalg.inv(scale_inv + scale_prior_inv)
        else:
            scale_in_hyper = 1 / (scale_inv + scale_prior_inv)
        scale_prior = wishart.rvs(df=gn, scale=scale_in_hyper,
                                  random_state=random_state)
    scale = scale_prior + .5 * (scaled_eps + scale_loc + scale_shape)

    return loc, shape, scale, w_loc_shape, degrees_of_freedom


def _structured_niw_rvs(loc, shape, scale, w_loc_shape, degrees_of_freedom, dim,
                        random_state=None):
    """Sample from the structured Normal-inverse-Wishart distribution.

    Parameters
    ----------

    loc : array of shape (n_features, )
        Location parameter.

    shape :  array of shape (n_features, )
        Shape parameter.

    scale : array of shape (n_features, n_features)
        Scale parameter.

    w_loc_shape : array of shape (2, 2)
        Diagonal matrix. The upper element controls the information
        in loc. The lower element controls the information in shape.

    degrees_of_freedom : float
        Degree of freedom parameter. Must be superior to n_features - 1.

    dim : int
        Dimension of the parameter space. Equal to n_features.

    random_state : boolean, default=None
        Optional, random seed used to initialize the pseudorandom number generator.
        If the random seed is None the np.random.RandomState singleton is used.


    Returns
    -------

    loc : array of shape (n_features, )

    shape : array of shape (n_features, )

    scale : array of shape (n_features, n_features)
    """

    scale = invwishart.rvs(df=degrees_of_freedom, scale=scale,
                           random_state=random_state)
    rns = multivariate_normal.rvs(size=dim * 2, random_state=random_state)
    loc_shape = np.hstack((loc, shape)) + \
                rns @ cholesky(np.kron(w_loc_shape, scale), lower=False)
    loc, shape = loc_shape[:dim], loc_shape[dim:]
    return loc, shape, scale


def _structured_niw_logpdf(loc_obs, shape_obs, scale_obs, loc, shape, scale,
                           w_loc_shape, degrees_of_freedom, allow_singular):
    """Structured Normal-inverse-Wishart log pdf.

    Parameters
    ----------

    loc_obs : array of shape (n_features, )
        Location observation.

    shape_obs :  array of shape (n_features, )
        Shape observation.

    scale_obs : array of shape (n_features, n_features)
        Scale observation.

    loc : array of shape (n_features, )
        Location parameter.

    shape :  array of shape (n_features, )
        Shape parameter.

    scale : array of shape (n_features, n_features)
        Scale parameter.

    w_loc_shape : array of shape (2, 2)
        Diagonal matrix. The upper element controls the information
        in loc. The lower element controls the information in shape.

    degrees_of_freedom : float
        Degree of freedom parameter. Must be superior to n_features - 1.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.


    Returns
    -------

    eval : float
        Evaluation of the pdf at the observations conditionally on parameters.
    """
    inv_w_pdf = invwishart.pdf(scale_obs, df=degrees_of_freedom, scale=scale)
    normal_logpdf = \
        multivariate_normal.logpdf(np.hstack((loc_obs, shape_obs)),
                                   mean=np.hstack((loc, shape)),
                                   cov=np.kron(w_loc_shape, scale_obs),
                                   allow_singular=allow_singular)

    return np.log(inv_w_pdf + _MACHINE_FLOAT_MIN) + normal_logpdf


class BayesianSkewtMixture(GibbsSamplingMixture):
    """Bayesian estimation of a skew-t mixture using a collapsed
    gibbs sampling scheme.

    This class allows to infer an approximate posterior distribution over
    the parameters of a skew-t mixture distribution. The effective number
    of components can be inferred from the data. The type of prior for the
    weights distribution implemented in this class is an infinite mixture
    model with the Dirichlet Process (using the Stick-breaking representation).
    A slice sampler for the Stick breaking approach to the Dirichlet process
    elegantly enables a finite number of centers to be sampled within
    each iteration of the MCMC.


    .. versionadded:: 0.1

    Parameters
    ----------

    max_iter : int, default=100
        The number of MCMC iterations to perform. The number of
        samplings will be equal to max_iter - burn_in.

    burn_in : int, default=20
        The number of MCMC burn-in iterations to perform.

    n_components_init : int
        Initial number of components.

    max_components : int
        Maximum number of components.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::
            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    loc_prior : float or array of shape (n_features,), default=None.
        The hyper parameter of the Gaussian prior on the location distribution.
        If it is None, it is set to the mean of X.

    shape_prior : float or array of shape (n_features,), default=None.
        The hyper parameter of the Gaussian prior on the shape distribution.
        If it is None, it is set to zero.

    scale_prior : float or array of shape (n_features, n_features), default=None.
        The hyper parameter of the inverse-Wishart prior on the scale distribution.
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X.

    degrees_of_freedom_prior : float, default=None.
        The hyper parameter degrees of freedom of the inverse-Wishart prior on
        the scale distribution.
        If it is None, it's set to `n_features`.

    w_loc_prior : float, default=100
        Controls the prior information in loc_prior.

    w_shape_prior : float, default=100
        Controls the prior information in shape_prior.

    alpha_a_prior : float, default=1e-3
        The hyper parameter shape of the gamma prior on the scale factor
        (alpha) in the Dirichlet process.

    alpha_b_prior : float, default=1e-3
        The hyper parameter inverse scale of the gamma prior on the scale factor
        alpha in the Dirichlet process.

    hierarchical_scale_prior : boolean
        Put a hierarchical Wishart distribution prior on the scale prior.
        Relaxes the influence of the scale prior. Setting it to false
        may result in a faster convergence of the mcmc although it enables
        a better exploration of parameter space.

    allow_singular : boolean
        If a matrix to be inverted happens to be singular, the pseudo-inverse
        is computed instead.

    random_state : boolean, default=None
        Optional, random seed used to initialize the pseudorandom number generator.
        If the random seed is None np.random.randint is used to generate a seed.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints each iteration step.
        If greater than 1 then it prints also the log posterior and the time
        needed for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    TODO

    Examples
    --------
    TODO
    --------

    References
    ----------
    .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           "Bayesian  Inference for Finite Mixtures of Univariate
           and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11 (2010), 317-36.

    .. [2] Boris P. Hejblum, Chariff Alkhassim, Raphael Gottardo, François Caron,
           and Rodolphe Thiébaut "Sequential Dirichlet Process Mixtures of
           Multivariate Skew-t distributions for Model-based Clustering for
           model-based clustering of flow cytometry data",
           The Annals of applied statistics, Volume 13, Number 1 (2019), 638-660

    .. [3] Neal, R. M. Slice sampling. The Annals of statistics 31 (2003), 705-767

    .. [4] Walker, S.G. "Sampling the Dirichlet mixture model with slices",
               Commun. Stat., Simul. Comput. 36 (2007), 45-54

    .. [5] Kalli, M., Griffin, J. E., and Walker, S. G. "Slice sampling mixture
           models", Statistics ans Computing 21 (2011), 93-105

    .. [6] Pitman, J. "Combinatorial Stochastic Processes", volume 1875 of Lecture
           Notes in Mathematics. Springer-Verlag, Berlin Heidelberg (2006).

    .. [7] Sethuraman, J. "A constructive definition of the Dirichlet priors."
           Statistica Sinica 4 (1994), 639-650

    .. [8] West, M. "hyper parameter estimation in Dirichlet process mixture
           models", In IDSD discussion paper series (1992), 92-03.
           Duke University.

    .. [9] Abramowitz, Milton, Stegun, Irene Ann.
           "Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables". Applied Mathematics Series. 1964, 949.


    """

    def __init__(self, max_iter=100, burn_in=20, n_components_init=20,
                 max_components=100, init_params='random',
                 loc_prior=None, shape_prior=None, scale_prior=None,
                 degrees_of_freedom_prior=None,
                 w_loc_prior=100, w_shape_prior=100,
                 alpha_a_prior=1e-3, alpha_b_prior=1e-3,
                 hierarchical_scale_prior=True, allow_singular=True,
                 cdf_approx=True, random_state=None,
                 verbose=0, verbose_interval=10):
        super().__init__(
            max_iter=max_iter, burn_in=burn_in,
            n_components_init=n_components_init,
            max_components=max_components,
            init_params=init_params,
            random_state=random_state,
            verbose=verbose, verbose_interval=verbose_interval)

        self.loc_prior = loc_prior
        self.shape_prior = shape_prior
        self.scale_prior = scale_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.w_loc_prior = w_loc_prior
        self.w_shape_prior = w_shape_prior
        self.alpha_a_prior = alpha_a_prior
        self.alpha_b_prior = alpha_b_prior

        self.hierarchical_scale_prior = hierarchical_scale_prior

        self.allow_singular = allow_singular

        self.cdf_approx = cdf_approx

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """

        self._check_loc_prior_parameter(X)
        self._check_shape_prior_parameter(X)
        self._check_scale_prior_parameter(X)
        self._check_degrees_of_freedom_prior(X)
        self._check_w_loc_shape_priors()
        self._check_alpha_priors()

    def _check_loc_prior_parameter(self, X):
        """Check the location hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.loc_prior is None:
            self.loc_prior_ = X.mean(axis=0)
        else:
            self.loc_prior_ = check_array(self.loc_prior,
                                          dtype=[np.float64, np.float32],
                                          ensure_2d=False)
            self._check_shape(self.loc_prior_, (dim,), 'loc')

    def _check_shape_prior_parameter(self, X):
        """Check the shape hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.shape_prior is None:
            self.shape_prior_ = np.zeros(shape=dim)
        else:
            self.loc_prior_ = check_array(self.shape_prior,
                                          dtype=[np.float64, np.float32],
                                          ensure_2d=False)
            self._check_shape(self.shape_prior_, (dim,), 'shape')

    def _check_scale_prior_parameter(self, X):
        """Check the scale hyper parameter of the normal-inverse-
        Whishart prior.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        _, dim = X.shape

        if self.scale_prior is None:
            self.scale_prior_ = np.cov(X.T)
        else:
            self.scale_prior_ = check_array(
                self.scale_prior, dtype=[np.float64, np.float32],
                ensure_2d=False)
            self._check_shape(self.scale_prior_, (dim, dim), 'scale')
            self._check_scale_matrix(self.scale_prior)
        # used for the Wishart hierarchical prior on the scale
        if dim > 1:
            self.scale_prior_inv_ = np.linalg.inv(self.scale_prior_)
        else:
            self.scale_prior_inv_ = 1 / self.scale_prior_

    def _check_w_loc_shape_priors(self):
        """Check the hyper parameters which control the prior information
        in loc_prior and shape_prior."""

        _check_positive_scalar(self.w_loc_prior, 'w_loc')
        _check_positive_scalar(self.w_shape_prior, 'w_weight')

        self.w_loc_shape_prior_ = np.diag([self.w_loc_prior,
                                           self.w_shape_prior])

    def _check_degrees_of_freedom_prior(self, X):
        """Check the hyper parameter degrees of freedom of the inverse-Wishart
        prior on the scale distribution."""

        _, dim = X.shape
        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = dim
        else:
            _check_positive_scalar(self.degrees_of_freedom_prior,
                                   'degrees_of_freedom_prior')
            if self.degrees_of_freedom_prior < dim:
                raise ValueError("degrees_of_freedom_prior should be, "
                                 "superior of equal to %s, but got %s, "
                                 "(dim, self.degrees_of_freedom_prior)")
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        # In 1d we shift the prior away from zero
        self.degrees_of_freedom_prior_ += 1 if dim == 1 else 0

    def _check_alpha_priors(self):
        """Check the hyper parameters of the gamma prior on the scale factor
        (alpha) in the Dirichlet process."""

        _check_positive_scalar(self.alpha_a_prior, 'alpha_a_prior')
        _check_positive_scalar(self.alpha_b_prior, 'alpha_b_prior')
        self.alpha_a_prior_ = self.alpha_a_prior
        self.alpha_b_prior_ = self.alpha_b_prior

    def _initialize(self, X, labels):
        """Initialization of the data storage structures and
        of the mixture parameters.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        labels : array of shape (n_observations, )
            The labels of each observation.
        """
        _, self.dim_ = X.shape

        self.labels_ = labels
        self.n_samples_ = len(labels)
        self.n_components_ = len(set(labels))
        self.n_samples_component_, _, self.ind_cl_ = \
            _clusters_sizes(labels, self.max_components)

        # skewt parameters
        self.locs_ = np.empty(shape=(self.max_components, self.dim_),
                              dtype=np.float64)
        self.shapes_ = np.empty(shape=(self.max_components, self.dim_),
                                dtype=np.float64)
        self.scales_ = \
            np.empty(shape=(self.max_components, self.dim_, self.dim_),
                     dtype=np.float64)
        self.scales_invs_ = \
            np.zeros(shape=(self.max_components, self.dim_, self.dim_),
                     dtype=np.float64)
        self.degrees_of_freedoms_ = \
            np.repeat(float(self.degrees_of_freedom_prior_), self.max_components)

        # structured normal-inverse-Wishart parameters
        # locs observations of the normal component
        self.sniw_locs_obs_ = np.empty(shape=(self.max_components, self.dim_),
                                       dtype=float)
        # locs of the normal component
        self.sniw_locs_ = np.empty(shape=(self.max_components, self.dim_),
                                   dtype=float)
        # shapes observations of the normal component
        self.sniw_shapes_obs_ = np.empty(shape=(self.max_components, self.dim_),
                                         dtype=float)
        # shapes of the normal component
        self.sniw_shapes_ = np.empty(shape=(self.max_components, self.dim_),
                                     dtype=float)
        # scales observations of the inverse-Wishart component
        self.sniw_scales_obs_ = np.empty(shape=(self.max_components, self.dim_,
                                                self.dim_),
                                         dtype=float)
        # scales of the inverse-Wishart component
        self.sniw_scales_ = np.empty(shape=(self.max_components, self.dim_, self.dim_),
                                     dtype=float)
        self.sniw_w_loc_shapes_ = np.empty(shape=(self.max_components, 2, 2),
                                           dtype=float)
        self.sniw_degrees_of_freedoms_ = np.empty(shape=self.max_components,
                                                  dtype=float)

        # random initialisation of latent truncated normal
        self.tn_ = _rtrunc_norm(mean=np.repeat(0, self.n_samples_),
                                sd=1, lower=0, upper=float('Inf'),
                                random_state=self.random_state_iter_)

        self.tn_ = np.repeat(float(1), self.n_samples_)
        # initialize scales
        self.scales_tn_ = np.repeat(float(1), self.n_samples_)
        # initialize alpha
        self.alpha_ = np.log(self.n_samples_)
        self._estimate_alpha_posterior()
        # structured Normal-inverse-Wishart posterior parameters estimation
        self._estimate_structured_niw_posterior_params(X)

    def _estimate_structured_niw_posterior_params(self, X):
        """Structured Normal-inverse-Wishart posterior
        sampling for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
        scales_invs = self.scales_invs_[self.ind_cl_].sum(axis=0)
        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            snpp = \
                _structured_niw_posterior_params(X[obs_k],
                                                 self.scales_tn_[obs_k],
                                                 self.loc_prior_,
                                                 self.shape_prior_,
                                                 self.scale_prior_,
                                                 self.w_loc_shape_prior_,
                                                 self.degrees_of_freedom_prior_,
                                                 self.tn_[obs_k],
                                                 self.hierarchical_scale_prior,
                                                 scales_invs,
                                                 self.scale_prior_inv_,
                                                 self.n_components_,
                                                 self.dim_,
                                                 self.allow_singular,
                                                 self.random_state_iter_)
            loc, shape, scale, w_loc_shape, degree_of_freedom = snpp
            # skewt parameters
            snr = _structured_niw_rvs(loc, shape, scale, w_loc_shape,
                                      degree_of_freedom, self.dim_,
                                      random_state=self.random_state_iter_)
            self.locs_[k], self.shapes_[k], self.scales_[k] = snr
            loc_sample, shape_sample, scale_sample = snr
            # structured normal-inverse-Wishart parameters (used to evaluate the posterior)
            # locs observations of the normal component
            self.sniw_locs_obs_[k] = loc_sample
            # locs of the normal component
            self.sniw_locs_[k] = loc
            # shapes observations of the normal component
            self.sniw_shapes_obs_[k] = shape_sample
            # shapes of the normal component
            self.sniw_shapes_[k] = shape
            # scales observations of the inverse-Wishart component
            self.sniw_scales_obs_[k] = scale_sample
            # scales of the inverse-Wishart component
            self.sniw_scales_[k] = scale
            self.sniw_w_loc_shapes_[k] = w_loc_shape
            self.sniw_degrees_of_freedoms_[k] = degree_of_freedom

    def _stick_breaking_slice_sampler(self, X):
        """Slice sampler for the Stick breaking approach to the Dirichlet process.
        Elegantly enables a finite number of centers to be sampled within
        each iteration of the MCMC.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Pitman, J. "Combinatorial Stochastic Processes", volume 1875 of Lecture
               Notes in Mathematics. Springer-Verlag, Berlin Heidelberg (2006).

        .. [2] Sethuraman, J. "A constructive definition of the Dirichlet priors."
               Statistica Sinica 4 (1994), 639-650

        .. [3] Neal, R. M. "Slice sampling", The Annals of statistics 31 (2003), 705-767

        .. [4] Walker, S.G. "Sampling the Dirichlet mixture model with slices",
               Commun. Stat., Simul. Comput. 36 (2007), 45-54

        .. [5] Kalli, M., Griffin, J. E., and Walker, S. G. "Slice sampling mixture
               models", Statistics ans Computing 21 (2011), 93-105

        """

        self.w_ = np.zeros(self.max_components, dtype=float)
        # sample the weights of each existing cluster from a Dirichlet distribution
        # and sample the rest of the weigth for potential new clusters
        # from a Gamma(alpha, 1) distribution.
        gamma_shapes = np.hstack((self.n_samples_component_[self.ind_cl_],
                                  self.alpha_ + _MACHINE_FLOAT_MIN))
        rgamma = gamma.rvs(a=gamma_shapes, size=len(gamma_shapes),
                           random_state=self.random_state_iter_)
        norm_rgamma = rgamma / rgamma.sum()
        self.w_[self.ind_cl_] = norm_rgamma[:-1]
        w_left = norm_rgamma[-1]  # weight left for potential new clusters
        # for each observation, sample a uniform random variable according to the weight
        # of its class.
        # the latent u is used in the slice sampling scheme
        self.u_ = uniform.rvs(size=self.n_samples_,
                              random_state=self.random_state_iter_) * \
                  self.w_[self.labels_]
        min_u = self.u_.min()
        # Sample the remaining weights that are needed with stick-breaking
        # i.e. the new clusters
        ind_potential_cl = np.nonzero(self.n_samples_component_ == 0)[0]  # potential new clusters
        card_new_cl = len(ind_potential_cl)

        if len(ind_potential_cl) and w_left > min_u:
            cpt_new_cl = 0  # number of new clusters
            if self.random_state_iter_ is not None:
                random_state_i = self.random_state_iter_
            else:
                random_state_i = np.random.randint(_RANDOM_STATE_UPPER)
            while w_left > min_u and cpt_new_cl < card_new_cl:
                rbeta = beta.rvs(a=1, b=self.alpha_,
                                 random_state=random_state_i)
                ind_new_cl = ind_potential_cl[cpt_new_cl]  # index of new cluster
                # weight of new cluster
                self.w_[ind_new_cl] = rbeta * w_left
                w_left = (1 - rbeta) * w_left
                random_state_i += 1
                # sample centers from prior
                loc, shape, scale = \
                    _structured_niw_rvs(self.loc_prior_,
                                        self.shape_prior_,
                                        self.scale_prior_,
                                        self.w_loc_shape_prior_,
                                        self.degrees_of_freedom_prior_,
                                        self.dim_,
                                        random_state=self.random_state_iter_)
                self.locs_[ind_new_cl] = loc
                self.shapes_[ind_new_cl] = shape
                self.scales_[ind_new_cl] = scale
                self.degrees_of_freedoms_[ind_new_cl] = \
                    self.degrees_of_freedom_prior_
                cpt_new_cl += 1

        temp_ind_cl = self.w_.nonzero()[0]
        clusters_pdfs = \
            np.empty(shape=(len(temp_ind_cl), self.n_samples_), dtype=float)
        for i, k in enumerate(temp_ind_cl):
            clusters_pdfs[i, :] = rv.pdf(X,
                                         mean=self.locs_[k],
                                         psi=self.shapes_[k],
                                         cov=self.scales_[k],
                                         df=self.degrees_of_freedoms_[k],
                                         allow_singular=self.allow_singular,
                                         cdf_approx=self.cdf_approx)
        # existing weights
        w_non_empty = self.w_[temp_ind_cl]
        # slices
        sliced_clusters = np.greater(w_non_empty[:, np.newaxis], self.u_)
        sliced_clusters_pdfs = clusters_pdfs * sliced_clusters
        sliced_clusters_probs = (sliced_clusters_pdfs / sliced_clusters_pdfs.
                                 sum(axis=0))
        sliced_clusters_cumuls = sliced_clusters_probs.cumsum(axis=0)
        # random_state_iter_ + 1 because variable u_ is sampled with random_state_iter_
        u_clust = uniform.rvs(size=self.n_samples_,
                              random_state=self.random_state_iter_ + 1)
        # newly sampled partition
        temp_labels = np.argmax(sliced_clusters_cumuls > u_clust, axis=0)
        # back to the original labels
        self.labels_ = temp_ind_cl[temp_labels]
        # number of observation in each cluster
        self.n_samples_component_, self.n_components_, self.ind_cl_ = \
            _clusters_sizes(self.labels_, self.max_components)

    def _estimate_truncated_normal_posterior(self, X):
        """Truncated normal posterior
        sampling of the random effects for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
        for k in self.ind_cl_:
            obs_k = np.where(self.labels_ == k)[0]
            loc_k = self.locs_[k]
            shape_k = self.shapes_[k]
            scale_k = self.scales_[k]
            psd_scale_k = PSD(scale_k, allow_singular=self.allow_singular)
            scale_k_inv = psd_scale_k.U @ np.transpose(psd_scale_k.U)
            self.scales_invs_[k] = scale_k_inv
            A_k = 1 / (1 + np.sum(np.square(np.dot(shape_k, psd_scale_k.U)), axis=-1))
            a_ik = A_k * ((X[obs_k] - loc_k) @ (shape_k @ scale_k_inv))
            self.tn_[obs_k] = \
                _rtrunc_norm(mean=a_ik, sd=np.sqrt(A_k / self.scales_tn_[obs_k]),
                             lower=0, upper=float('inf'),
                             random_state=self.random_state_iter_)

    def _estimate_truncated_normal_posterior_1d(self, X):
        """Truncated normal posterior
        sampling of the random effects for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
        for k in self.ind_cl_:
            # observations in cluster k
            obs_k = np.where(self.labels_ == k)[0]
            loc_k = self.locs_[k]
            shape_k = self.shapes_[k]
            scale_k = self.scales_[k]
            scale_k_inv = 1 / scale_k
            self.scales_invs_[k] = scale_k_inv
            A_k = 1 / (1 + shape_k ** 2 * scale_k_inv)
            a_ik = (A_k * ((X[obs_k] - loc_k) * (shape_k * scale_k_inv)))[:, 0]
            self.tn_[obs_k] = \
                _rtrunc_norm(mean=a_ik, sd=np.sqrt(A_k / self.scales_tn_[obs_k]),
                             lower=0, upper=float('inf'),
                             random_state=self.random_state_iter_)

    def _estimate_alpha_posterior(self):
        """Posterior sampling of the Gamma prior on the scale
        parameter of the Dirichlet process.

        References
        ----------

        .. [1] West, M. "hyper parameter estimation in Dirichlet process mixture
           models", In IDSD discussion paper series (1992), 92-03.
           Duke University.

        """
        x = beta.rvs(a=self.alpha_ + 1, b=self.n_samples_,
                     random_state=self.random_state_iter_)
        pi_det = (self.alpha_a_prior_ + self.n_components_ - 1)
        pi_num = (self.n_samples_ *
                  (self.alpha_b_prior_ - np.log(x + _MACHINE_FLOAT_MIN)))
        pi = pi_det / pi_num
        pi /= 1 + pi
        if uniform.rvs(random_state=self.random_state_iter_) < pi:
            self.alpha_ = \
                gamma.rvs(a=self.alpha_a_prior_ + self.n_components_,
                          scale=1 / (self.alpha_b_prior_ -
                                     np.log(x + _MACHINE_FLOAT_MIN)),
                          random_state=self.random_state_iter_)
        else:
            self.alpha_ = \
                gamma.rvs(a=self.alpha_a_prior_ + self.n_components_ - 1,
                          scale=1 / (self.alpha_b_prior_ -
                                     np.log(x + _MACHINE_FLOAT_MIN)),
                          random_state=self.random_state_iter_)

    def _estimate_degree_of_freedom_posterior(self, X):
        """Metropolis-Hasting within Gibbs to sample the degree of freedom.
        The structured Normal-inverse-Wishart is a conjugate prior
        of a skew-t distribution for all its parameters except for the
        degree of freedom.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
        card_clusters = len(self.ind_cl_)
        clusters_pdfs_old = \
            np.zeros(shape=(card_clusters, self.n_samples_), dtype=float)
        clusters_pdfs_new = \
            np.zeros(shape=(card_clusters, self.n_samples_), dtype=float)
        # used to compute the log posterior in _log_posterior_eval
        self.clusters_pdfs_ = \
            np.zeros(shape=(card_clusters, self.n_samples_), dtype=float)
        degrees_of_freedoms_new = np.empty(card_clusters, dtype=float)
        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            degree_of_freedoms_k_old = self.degrees_of_freedoms_[k]
            clusters_pdfs_old[i, obs_k] = \
                rv.logpdf(X[obs_k],
                          mean=self.locs_[k],
                          psi=self.shapes_[k],
                          cov=self.scales_[k],
                          df=degree_of_freedoms_k_old,
                          allow_singular=self.allow_singular,
                          cdf_approx=self.cdf_approx)

            runif_a = \
                np.log(degree_of_freedoms_k_old - 1 + _MACHINE_FLOAT_MIN) -\
                _C_DF_MH
            runif = uniform.rvs(loc=runif_a, scale=2 * _C_DF_MH,
                                random_state=self.random_state_iter_)

            degree_of_freedom_k_new = 1 + np.exp(runif)
            degrees_of_freedoms_new[i] = degree_of_freedom_k_new
            clusters_pdfs_new[i, obs_k] = \
                rv.logpdf(X[obs_k],
                          mean=self.locs_[k],
                          psi=self.shapes_[k],
                          cov=self.scales_[k],
                          df=degree_of_freedom_k_new,
                          allow_singular=self.allow_singular,
                          cdf_approx=self.cdf_approx)
        runifs = uniform.rvs(size=card_clusters,
                             random_state=self.random_state_iter_)
        if card_clusters > 1:
            indexes_to_slice = list(range(card_clusters))
            for i, k in enumerate(self.ind_cl_):
                i_ = indexes_to_slice[:i] + indexes_to_slice[i + 1:]

                num1_1 = clusters_pdfs_old[i_, :].sum() + \
                         clusters_pdfs_new[i, :].sum()
                num1_2 = log_jeffrey_prior(degrees_of_freedoms_new[i])
                num1_3 = np.log(degrees_of_freedoms_new[i] - 1 +
                                _MACHINE_FLOAT_MIN)
                den1_1 = clusters_pdfs_old.sum()
                den1_2 = log_jeffrey_prior(self.degrees_of_freedoms_[k])
                den1_3 = np.log(self.degrees_of_freedoms_[k] - 1 +
                                _MACHINE_FLOAT_MIN)
                with np.errstate(over='ignore'):
                    prob_transition = np.exp((num1_1 + num1_2 + num1_3 -
                                              den1_1 - den1_2 - den1_3))
                if runifs[i] < min(1, prob_transition):
                    self.degrees_of_freedoms_[k] = degrees_of_freedoms_new[i]
                    # faster than deepcopying clusters_pdfs_old
                    self.clusters_pdfs_[i, :] = clusters_pdfs_new[i, :]
                else:
                    self.clusters_pdfs_[i, :] = clusters_pdfs_old[i, :]

        else:
            num1_1 = clusters_pdfs_new.sum()
            num1_2 = log_jeffrey_prior(degrees_of_freedoms_new)
            num1_3 = np.log(degrees_of_freedoms_new - 1 +
                            _MACHINE_FLOAT_MIN)
            den1_1 = clusters_pdfs_old.sum()
            den1_2 = log_jeffrey_prior(degree_of_freedoms_k_old)
            den1_3 = np.log(degree_of_freedoms_k_old - 1 +
                            _MACHINE_FLOAT_MIN)

            with np.errstate(over='ignore'):
                prob_transition = np.exp((num1_1 + num1_2 + num1_3 -
                                          den1_1 - den1_2 - den1_3))

            if runifs < min(1, prob_transition):
                self.degrees_of_freedoms_[self.ind_cl_[0]] = \
                    degrees_of_freedoms_new[0]
                # faster than deepcopying clusters_pdfs_old
                self.clusters_pdfs_[i, :] = clusters_pdfs_new[i, :]
            else:
                self.clusters_pdfs_[i, :] = clusters_pdfs_old[i, :]

    def _estimate_scale_posterior(self, X):
        """Gamma posterior sampling of scales
        of the truncated normal regressors for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """

        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            eps_k = X[obs_k, :] - self.locs_[k] - self.tn_[obs_k, np.newaxis] *\
                    self.shapes_[k][np.newaxis, :]
            trace_k = np.trace(eps_k[:, :, np.newaxis] @ eps_k[:, np.newaxis] @
                               self.scales_invs_[k], axis1=2, axis2=1)
            self.scales_tn_[obs_k] = \
                gamma.rvs(a=(self.degrees_of_freedoms_[k] + self.dim_ + 1) / 2,
                          scale=2 / (self.degrees_of_freedoms_[k] +
                                     self.tn_[obs_k] ** 2 + trace_k),
                          random_state=self.random_state_iter_)

    def _estimate_scale_posterior_1d(self, X):
        """Gamma posterior sampling of scales
        of the truncated normal regressors for each component.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        References
        ----------
        .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

        """
        for i, k in enumerate(self.ind_cl_):
            obs_k = np.where(self.labels_ == k)[0]
            eps_k = \
                X[obs_k, 0] - self.locs_[k] - self.tn_[obs_k] * self.shapes_[k]
            trace_k = eps_k ** 2 * self.scales_invs_[k]
            self.scales_tn_[obs_k] = \
                gamma.rvs(a=(self.degrees_of_freedoms_[k] + self.dim_ + 1) / 2,
                          scale=2 / (self.degrees_of_freedoms_[k] +
                                     self.tn_[obs_k] ** 2 + trace_k),
                          random_state=self.random_state_iter_)

    def _partition_sampling(self, X):
        """Posterior sampling of a partition.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """
        self._stick_breaking_slice_sampler(X)

    def _parameters_sampling(self, X):
        """Posterior sampling of a parameters set.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """

        if self.dim_ == 1:
            self._estimate_truncated_normal_posterior_1d(X)
            self._estimate_degree_of_freedom_posterior(X)
            self._estimate_scale_posterior_1d(X)

        else:
            self._estimate_truncated_normal_posterior(X)
            self._estimate_degree_of_freedom_posterior(X)
            self._estimate_scale_posterior(X)

        self._estimate_structured_niw_posterior_params(X)
        self._estimate_alpha_posterior()

    def _sampled_partition(self):
        """Return a sample from the dirichlet process skew-t
        mixture posterior distribution"""
        return self.labels_

    def _sampled_parameters(self):
        """ sampled parameters for each component of the
        skew-t mixture.

        Returns
        -------

        A dictionary containing dictionaries:
            locs : k arrays of shape (n_features, )
            shapes : k arrays of shape (n_features, )
            scales : k arrays of shape (n_features, n_features)
            degrees_of_freedoms : k floats
        """
        locs, shapes, scales, dfs = {}, {}, {}, {}
        for k in self.ind_cl_:
            locs[k] = self.locs_[k]
            shapes[k] = self.shapes_[k]
            scales[k] = self.scales_[k]
            dfs[k] = self.degrees_of_freedoms_[k]

        return {'locs': locs, 'shapes': shapes,
                'scales': scales, 'degrees_of_freedoms': dfs}

    def _map_predict(self, X, map_params, map_labels):
        """Predict the labels for the data samples in X using the MAP
        of the trained model.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        map_labels : array of shape (n_observations, )
            MAP partition.

        map_params : dict of size n_components
            MAP parameters.

        Returns
        -------
        labels : array of shape (n_observations, )
            Predicted partition.

        Note
        ------
        Due to the nature of the slice sampling, this function may be slightly unstable.
        For instance. If one tries to predict the same data used for training, a
        slightly different partition may result.
        """
        map_locs = map_params['locs']
        map_shapes = map_params['shapes']
        map_scales = map_params['scales']
        map_dfs = map_params['degrees_of_freedoms']

        n_observations, _ = X.shape
        n_components = len(map_params)
        clusters_pdfs = np.zeros(shape=(n_components, n_observations), dtype=float)
        set_map_labels = np.empty(n_components, dtype=int)
        for i, (key, map_loc) in enumerate(map_locs.items()):
            clusters_pdfs[i, :] = rv.pdf(X, mean=map_loc,
                                         psi=map_shapes[key],
                                         cov=map_scales[key],
                                         df=map_dfs[key],
                                         allow_singular=self.allow_singular)
            set_map_labels[i] = key
        return set_map_labels[np.argmax(clusters_pdfs, axis=0)]

    def _log_posterior_eval(self):
        """Evaluate the log posterior.

        Returns
        -------

        A dictionary containing:
            mixture: float
                observed log likelihood ot the mixture
            clustering : float
                log likelihood of the clustering
            sniw_prior : float
                log likelihood of the structured Normal-Inverse-Wishart prior
            gamma_prior: float
                log likelihood of the gamma prior on the scale of the Dirichlet
                process.
        """
        # observed skew-t mixture log likelihood
        n_samples_component = self.n_samples_component_[self.ind_cl_]
        log_mixing = np.log(n_samples_component / self.n_samples_)
        log_lik_mixture = (self.clusters_pdfs_.sum(axis=1) +
                           n_samples_component * log_mixing).sum()
        # Chinese Restaurant Process log likelihood (clustering log likelihood)
        part0 = loggamma(self.alpha_ + _MACHINE_FLOAT_MIN)
        part1 = self.n_components_ * np.log(self.alpha_ + _MACHINE_FLOAT_MIN)
        part2 = loggamma(n_samples_component).sum()
        part3 = loggamma(self.alpha_ + self.n_samples_)
        log_lik_clustering = part0 + part1 + part2 - part3

        # normal-inverse-Wishart log likelihood
        log_lik_sniw_prior = float()
        for k in self.ind_cl_:
            loc_obs = self.sniw_locs_obs_[k]
            loc = self.sniw_locs_[k]
            shape_obs = self.sniw_shapes_obs_[k]
            shape = self.sniw_shapes_[k]
            scale_obs = self.sniw_scales_obs_[k]
            scale = self.sniw_scales_[k]
            w_loc_shape = self.sniw_w_loc_shapes_[k]
            degree_of_freedom = self.sniw_degrees_of_freedoms_[k]
            log_lik_sniw_prior += \
                _structured_niw_logpdf(loc_obs, shape_obs,
                                       scale_obs, loc, shape,
                                       scale, w_loc_shape,
                                       degree_of_freedom,
                                       allow_singular=self.allow_singular)
        # gamma prior log likelihood
        log_lik_gamma_prior = gamma.logpdf(self.alpha_ + _MACHINE_FLOAT_MIN,
                                           a=self.alpha_a_prior_,
                                           scale=self.alpha_b_prior_)

        return {'mixture': log_lik_mixture,
                'crp': log_lik_clustering,
                'sniw_prior': log_lik_sniw_prior,
                'gamma_prior': log_lik_gamma_prior}