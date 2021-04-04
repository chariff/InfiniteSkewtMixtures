# Author: Chariff Alkhassim <chariff.alkhassim@gmail.com>
# License: MIT

from time import time

import numpy as np

from sklearn.utils import check_array
from sklearn.cluster import KMeans

from abc import ABCMeta, abstractmethod

import warnings

_RANDOM_STATE_UPPER = 1e5


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array
    param_shape : tuple
    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.
    Parameters
    ----------
    X : array of shape (n_observations, n_features)
    n_components : int

    Returns
    -------
    X : array of shape (n_observations, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_observations >= n_components '
                         'but got n_components = %d, n_observations = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    if X.shape[0] <= X.shape[1]:
        raise ValueError("Expected the number of observations to be "
                         "superior to the number of features, "
                         "but got %d features and %d observations"
                         % (X.shape[1], X.shape[0]))
    return X


def _check_random_state(random_state):
    """Check the input random seed.

    Returns
    -------
    random_seed : int
    """
    if random_state is None:
        return np.random.randint(_RANDOM_STATE_UPPER)
    else:
        return random_state


class GibbsSamplingMixture(metaclass=ABCMeta):
    """Gibbs sampling base class for mixture models.

    This abstract class specifies an interface for all mixture
    classes and provides basic common methods for mixture models.
    """

    def __init__(self, max_iter, burn_in, n_components_init, max_components,
                 init_params, random_state, verbose, verbose_interval):

        self.max_iter = max_iter
        self.burn_in = burn_in
        self.n_components_init = n_components_init
        self.max_components = max_components
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        # initialization is done in the _initialize_fit_storage method
        self.logposteriors_evals = None
        self.logposteriors = None
        self.random_state_iter_ = None
        self.sampled_partitions = None
        self.sampled_parameters = None

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
        """
        n_observations, _ = X.shape

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)
        if self.burn_in >= self.max_iter:
            raise ValueError("Invalid value for 'burn_in': %d "
                             "burn_in must be inferior to max_iter"
                             % self.max_iter)

        if self.n_components_init < 1:
            raise ValueError("Invalid value for 'n_components_init': %d "
                             "Estimation requires at least one component"
                             % self.n_components_init)
        elif self.n_components_init > n_observations:
            raise ValueError("Invalid value for 'n_components_init': %d "
                             "Number of components must be inferior to "
                             "the number of observations"
                             % self.n_components_init)

        if self.max_components < self.n_components_init:
            raise ValueError("Invalid value for 'max_components': %d "
                             "Maximum number of components must be superior or "
                             "equal to 'n_components_init'"
                             % self.max_components)
        elif self.max_components > n_observations:
            raise ValueError("Invalid value for 'max_components': %d "
                             "Maximum number of components must be inferior or "
                             "equal to the number of observations"
                             % self.max_components)

        # Check all the parameters values of the derived class
        self._check_parameters(X)

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array of shape  (n_observations, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_observations, _ = X.shape

        if self.init_params == 'kmeans':
            labels = KMeans(n_clusters=self.n_components_init,
                            n_init=1, random_state=random_state).fit(X).labels_
        elif self.init_params == 'random':
            random_state = np.random.RandomState(random_state)
            labels = random_state.choice(np.arange(self.n_components_init),
                                         n_observations)
            k_random = len(set(labels))
            if k_random < self.n_components_init:
                warnings.warn("The randomly generated partition "
                              "has less components than n_components_init. "
                              "Got %s instead of %s"
                              % (k_random, self.n_components_init))
        elif self.init_params == 'partition':
            # TODO enables a prior partition
            raise NotImplementedError
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

        self._initialize(X, labels)

    def _initialize_fit_storage(self, X, random_state):
        """Initialize storage structures used in the fit method.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_observations, _ = X.shape
        n_samplings = self.max_iter - self.burn_in
        self.logposteriors_evals = {}
        self.logposteriors = np.empty(self.max_iter, dtype=float)
        self.sampled_partitions = np.empty(shape=(n_samplings, n_observations),
                                           dtype=int)
        self.sampled_parameters = {}
        self.random_state_iter_ = random_state

    @abstractmethod
    def _initialize(self, X, labels):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_observations, n_features)

        labels : array of shape (n_observations, )
            The labels of each observation.
        """
        pass

    @abstractmethod
    def _partition_sampling(self, X):
        """Posterior sampling of a partition.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """
        pass

    @abstractmethod
    def _parameters_sampling(self, X):
        """Posterior sampling of a parameters set.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        """
        pass

    @abstractmethod
    def _sampled_partition(self):
        """A sampled partition from the posterior."""
        pass

    @abstractmethod
    def _sampled_parameters(self):
        """Sets of sampled parameters from the posterior
        for each component of the mixture."""
        pass

    @abstractmethod
    def _log_posterior_eval(self):
        """Evaluate the log posterior."""
        pass

    def fit(self, X):
        """Generate samples from the posterior distribution with
        a Gibbs sampling algorithm.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """

        n_observations, _ = X.shape
        X = _check_X(X, None, ensure_min_samples=2)
        self._check_initial_parameters(X)

        random_state = _check_random_state(self.random_state)

        self._initialize_fit_storage(X, random_state)
        self._initialize_parameters(X, random_state)

        n_burn_in = self.burn_in
        logposterior = -np.infty

        # burn-in period
        self._print_verbose_msg_beg('burn-in')

        for n_iter in range(n_burn_in):
            self.random_state_iter_ += n_iter

            self._partition_sampling(X)
            self._parameters_sampling(X)

            # store log posterior evaluation
            logposterior_eval = self._log_posterior_eval()
            self.logposteriors_evals[n_iter] = logposterior_eval
            logposterior = sum(logposterior_eval.values())
            self.logposteriors[n_iter] = logposterior
            self._print_verbose_msg_iter_end(n_iter, logposterior, 'Burn-in')

        self._print_verbose_msg_end(logposterior, 'burn-in')

        # sampling period
        self._print_verbose_msg_beg('sampling')
        n_samplings = self.max_iter - n_burn_in
        for n_iter in range(n_samplings):
            self.random_state_iter_ += n_iter

            self._partition_sampling(X)
            self._parameters_sampling(X)

            # store sampled partition
            self.sampled_partitions[n_iter] = self._sampled_partition()
            # store sampled parameters
            self.sampled_parameters[n_iter] = self._sampled_parameters()
            # store log posterior evaluation
            logposterior_eval = self._log_posterior_eval()
            self.logposteriors_evals[n_iter + n_burn_in] = logposterior_eval
            logposterior = sum(logposterior_eval.values())
            self.logposteriors[n_iter + n_burn_in] = logposterior
            self._print_verbose_msg_iter_end(n_iter, logposterior, 'Sampling')

        self._print_verbose_msg_end(logposterior, 'sampling')

        return self

    @abstractmethod
    def _map_predict(self, X, map_params, map_labels):
        """Predict the labels for the data samples in X using trained model.

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
        """
        pass

    def map_predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array of shape (n_observations, n_features)

        Returns
        -------
        labels : array of shape (n_observations, )
            Predicted partition.
        """
        if not hasattr(self, "sampled_partitions"):
            raise AttributeError("Unfitted model.")
        X = _check_X(X, None, ensure_min_samples=2)
        n_samplings = self.max_iter - self.burn_in
        map_ind = np.argmax(self.logposteriors[-n_samplings:])
        map_params = self.sampled_parameters[map_ind]
        map_labels = self.sampled_partitions[map_ind]
        return self._map_predict(X, map_params, map_labels)

    @property
    def partitions(self):
        """Sampled partitions.

        Returns
        -------
        partitions : array of shape (n_obsevations, n_samples)
            n_samples is equal to max_iter - n_burn_in.
        """
        if not hasattr(self, "sampled_partitions"):
            raise AttributeError("Unfitted model.")
        return self.sampled_partitions

    @property
    def parameters(self):
        """Sampled parameters
        for each component of the mixture.

        Returns
        -------
        parameters : dictionary of size (n_samples)
            n_samples is equal to max_iter - n_burn_in.
        """
        if not hasattr(self, "sampled_parameters"):
            raise AttributeError("Unfitted model.")
        return self.sampled_parameters

    @property
    def map_partition(self):
        """Maximum a posteriori partition.

        Returns
        -------
        partition : array of shape (n_obsevations, )
        """
        if not hasattr(self, "sampled_partitions"):
            raise AttributeError("Unfitted model.")
        n_samplings = self.max_iter - self.burn_in
        map_ind = np.argmax(self.logposteriors[-n_samplings:])
        return self.sampled_partitions[map_ind]

    @property
    def map_parameters(self):
        """Maximum a posteriori parameters.

        Returns
        -------
        parameters : dictionary
        map parameters for each component of the mixture
        """
        if not hasattr(self, "sampled_parameters"):
            raise AttributeError("Unfitted model.")
        n_samplings = self.max_iter - self.burn_in
        map_ind = np.argmax(self.logposteriors[-n_samplings:])
        return self.sampled_parameters[map_ind]

    @property
    def log_posterior_evals(self):
        """Log posterior evaluations for each component
        of the posterior for each iteration of the Gibbs sampler.

        Returns
        -------
        posterior_evals :  dictionary of size max_iter
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
        if not hasattr(self, "sampled_parameters"):
            raise AttributeError("Unfitted model.")
        return self.logposteriors_evals

    @property
    def log_posteriors(self):
        """Log posterior evaluation for each iteration
        of the Gibbs sampler.

        Returns
        -------
        """
        if not hasattr(self, "sampled_parameters"):
            raise AttributeError("Unfitted model.")
        return self.logposteriors

    def _print_verbose_msg_beg(self, case):
        """Print verbose message on beginning case."""
        if self.verbose == 1:
            print('Begin %s' % case)
        elif self.verbose >= 2:
            print('Begin %s' % case)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time

    def _print_verbose_msg_iter_end(self, n_iter, log_posterior, case):
        """Print verbose message on initialization."""
        if not n_iter % self.verbose_interval:
            if self.verbose == 1:
                print("%s iteration %d" % (case, n_iter))
            elif self.verbose >= 2:
                cur_time = time()
                print("%s iteration %d\t time lapse %.5fs\t log posterior %.5f" % (
                    case, n_iter, cur_time - self._iter_prev_time, log_posterior))
                self._iter_prev_time = cur_time

    def _print_verbose_msg_end(self, log_posterior, case):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("End %s" % case)
        elif self.verbose >= 2:
            print("End %s\t time lapse %.5fs\t log posterior %.5f" %
                  (case, time() - self._init_prev_time, log_posterior))
