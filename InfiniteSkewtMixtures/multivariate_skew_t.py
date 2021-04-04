# Author: Chariff Alkhassim <chariff.alkhassim@gmail.com>
# License: MIT

import numpy as np

import scipy.linalg
from scipy.stats import (t, gamma, multivariate_normal, uniform, norm)
from scipy.special import (loggamma, ndtr, ndtri)
from scipy._lib._util import check_random_state


_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)
_MACHINE_FLOAT_EPS = np.finfo('float').eps


# Author: Joris Vankerschaver
# License: BSD
def _squeeze_output(out):
    """
    Remove single-dimensional entries from array and convert to scalar,
    if necessary.
    """
    out = out.squeeze()
    if out.ndim == 0:
        out = out[()]
    return out


# Author: Joris Vankerschaver
# License: BSD
def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    """
    Determine which eigenvalues are "small" given the spectrum.
    This is for compatibility across various linear algebra functions
    that should agree about whether or not a Hermitian matrix is numerically
    singular and what is its numerical matrix rank.
    This is designed to be compatible with scipy.linalg.pinvh.
    Parameters
    ----------
    spectrum : 1d ndarray
        Array of eigenvalues of a Hermitian matrix.
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    Returns
    -------
    eps : float
        Magnitude cutoff for numerical negligibility.
    """
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps


# Author: Joris Vankerschaver
# License: BSD
def _pinv_1d(v, eps=1e-5):
    """
    A helper function for computing the pseudoinverse.
    Parameters
    ----------
    v : iterable of numbers
        This may be thought of as a vector of eigenvalues or singular values.
    eps : float
        Values with magnitude no greater than eps are considered negligible.
    Returns
    -------
    v_pinv : 1d float ndarray
        A vector of pseudo-inverted numbers.
    """
    return np.array([0 if abs(x) <= eps else 1 / x for x in v], dtype=float)


# Author: Joris Vankerschaver
# License: BSD
class PSD:
    """
    Compute coordinated functions of a symmetric positive semidefinite matrix.
    This class addresses two issues.  Firstly it allows the pseudoinverse,
    the logarithm of the pseudo-determinant, and the rank of the matrix
    to be computed using one call to eigh instead of three.
    Secondly it allows these functions to be computed in a way
    that gives mutually compatible results.
    All of the functions are computed with a common understanding as to
    which of the eigenvalues are to be considered negligibly small.
    The functions are designed to coordinate with scipy.linalg.pinvh()
    but not necessarily with np.linalg.det() or with np.linalg.matrix_rank().
    Parameters
    ----------
    M : ndarray
        Symmetric positive semidefinite matrix (2-D).
    cond, rcond : float, optional
        Cutoff for small eigenvalues.
        Singular values smaller than rcond * largest_eigenvalue are
        considered zero.
        If None or -1, suitable machine precision is used.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower
        or upper triangle of M. (Default: lower)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination) if the inputs do contain
        infinities or NaNs.
    allow_singular : bool, optional
        Whether to allow a singular matrix.  (Default: True)
    Notes
    -----
    The arguments are similar to those of scipy.linalg.pinvh().
    """

    def __init__(self, M, cond=None, rcond=None, lower=True,
                 check_finite=True, allow_singular=True):
        # Compute the symmetric eigendecomposition.
        # Note that eigh takes care of array conversion, chkfinite,
        # and assertion that the matrix is square.
        s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

        eps = _eigvalsh_to_eps(s, cond, rcond)
        if np.min(s) < -eps:
            raise ValueError('the input matrix must be positive semidefinite')
        d = s[s > eps]
        if len(d) < len(s) and not allow_singular:
            raise np.linalg.LinAlgError('singular matrix')
        s_pinv = _pinv_1d(s, eps)
        U = np.multiply(u, np.sqrt(s_pinv))

        # Initialize the eagerly precomputed attributes.
        self.rank = len(d)
        self.U = U
        self.log_pdet = np.sum(np.log(d))

        # Initialize an attribute to be lazily computed.
        self._pinv = None

    @property
    def pinv(self):
        if self._pinv is None:
            self._pinv = np.dot(self.U, self.U.T)
        return self._pinv


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

    ulower = ndtr((lower - mean) / sd)
    uupper = ndtr((upper - mean) / sd)
    draws = uniform.rvs(size=len(ulower), random_state=random_state)
    u = (uupper - ulower) * draws + ulower
    # prevent ndtriu from returning inf (integral over R)
    u[np.isclose(a=u, b=1, rtol=0, atol=1e-18, equal_nan=False)] = 1 - _MACHINE_FLOAT_EPS
    return mean + sd * ndtri(u)


class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.
    """

    def __init__(self, seed=None):
        super(multi_rv_generic, self).__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the RandomState object for generating random variates.
        This can be either None, int, a RandomState instance, or a
        np.random.Generator instance.
        If None (or np.random), use the RandomState singleton used by
        np.random.
        If already a RandomState or Generator instance, use it.
        If an int, use a new RandomState instance seeded with seed.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):
        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state


class multi_rv_frozen:
    """
    Class which encapsulates common functionality between all frozen
    multivariate distributions.
    """

    @property
    def random_state(self):
        return self._dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self._dist._random_state = check_random_state(seed)


class multivariate_skew_t_gen(multi_rv_generic):
    """
    A multivariate skew-t random variable.
    The `mean` keyword specifies the mean. The `cov` keyword specifies the
    covariance matrix.
    Methods
    -------

    pdf(x, mean=None, psi=None, cov=1 df=None, allow_singular=False)
        Probability density function.
    logpdf(x, mean=None, psi=None, cov=1, df=None, allow_singular=False)
        Log of the probability density function.
    rvs(mean=None, psi=None, cov=1, df=None, size=1, random_state=None)``
        Draw random samples from a multivariate skew-t distribution.

    Parameters
    ----------
    x : ndarray
        Quantiles, with the last axis of `x` denoting the components.

    Alternatively, the object may be called (as a function) to fix the mean,
    the shape `psi`, the covariance and the degree of freedom parameters,
    returning a "frozen" multivariate skew-t random variable:
    rv = multivariate_skew_t(mean=None, psi=None, cov=1, df=None, allow_singular=False)
        - Frozen object with the same methods but holding the given
          mean , shape, covariance and degree of freedom fixed.
    Notes
    -----

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    References
    ----------
    .. [1] Sylvia Fruhwirth-Schnatter and Saumyadipta Pyne
           Supplementary Material to "Bayesian  Inference for Finite Mixtures
           of Univariate and Multivariate Skew Normal and Skew-t Distributions",
           Biostatistics, 11, 317-36.

    .. versionadded:: 0.0.1
    Examples
    --------
    TODO
    """

    def __init__(self, seed=None):
        super(multivariate_skew_t_gen, self).__init__(seed)

    def __call__(self, mean=None, psi=None, cov=1, df=None,
                 allow_singular=False, seed=None):
        """
        Create a frozen multivariate skew-t distribution.
        See `multivariate_skew_t_frozen` for more information.
        """
        return multivariate_skew_t_frozen(mean, psi, cov, df,
                                          allow_singular=allow_singular,
                                          seed=seed)

    def _process_parameters(self, dim, mean, psi, cov, df):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean, psi and covariance are full vector resp. matrix.
        """

        # Try to infer dimensionality
        if dim is None:
            if mean is None:
                if cov is None:
                    dim = 1
                else:
                    cov = np.asarray(cov, dtype=float)
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
            else:
                mean = np.asarray(mean, dtype=float)
                dim = mean.size
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be "
                                 "a scalar.")
        # Check degree of freedom
        if df is None:
            df = 1
        elif not np.isscalar(df):
            raise ValueError("Degree of freedom must be a scalar.")
        # Check input sizes and return full arrays for mean, psi and cov if
        # necessary
        if mean is None:
            mean = np.zeros(dim)
        mean = np.asarray(mean, dtype=float)
        if psi is None:
            psi = np.zeros(dim)
        psi = np.asarray(psi, dtype=float)
        if cov is None:
            cov = 1.0
        cov = np.asarray(cov, dtype=float)

        if dim == 1:
            mean.shape = (1,)
            psi.shape = (1,)
            cov.shape = (1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." %
                             dim)

        if psi.ndim != 1 or psi.shape[0] != dim:
            raise ValueError("Array 'psi' must be a vector of length %d." %
                             dim)
        if psi.ndim == 1 and dim > 1:
            psi = np.copy(psi)
            psi.shape = (dim, 1)
        if cov.ndim == 0:
            cov = cov * np.eye(dim)
        elif cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim == 2 and cov.shape != (dim, dim):
            rows, cols = cov.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(cov.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(cov.shape), len(mean))
            raise ValueError(msg)
        elif cov.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % cov.ndim)
        return dim, mean, cov, psi, df

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]

        return x

    def _logpdf(self, x, mean, psi, cov, prec_U, df,
                log_det_cov, rank, dim, cdf_approx):
        """
        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function
        mean : ndarray
            Mean of the distribution.
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        prec_U : ndarray
            A decomposition such that np.dot(prec_U, prec_U.T)
            is the precision matrix, i.e. inverse of the covariance matrix.
        df : float
            Degree of freedom of the distribution.
        log_det_cov : float
            Logarithm of the determinant of the covariance matrix.
        rank : int
            Rank of the covariance matrix.
        dim : int
            dimension of the pdf.
        cdf_approx : bool, optional
            Whether to compute an approximation of the student cdf.
        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.
        """
        # log skew-t pdf part 1 (t log pdf)
        t_pdf_part1 = loggamma(df / 2) + dim / 2 * (_LOG_PI + np.log(df))
        t_pdf_part2 = loggamma((df + dim) / 2) - .5 * log_det_cov
        dev = x - mean
        maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        t_pdf_part3 = (df + dim) / 2 * np.log(1 + maha / df)
        t_pdf = t_pdf_part2 - t_pdf_part1 - t_pdf_part3
        # log skew-t pdf part 2 (t log cdf)
        psi_t = np.transpose(psi)
        omega = cov + psi @ psi_t
        small_omega_diag = np.sqrt(np.diag(omega))
        small_omega = np.diag(small_omega_diag)
        small_omega_inv = np.diag(1 / small_omega_diag)
        eta_num = small_omega @ prec_U @ np.transpose(prec_U) @ psi
        eta_den = np.sqrt(1 - np.sum(np.square(np.dot(psi_t, prec_U))))
        eta = eta_num / eta_den
        t_cdf_q_part1 = np.squeeze(np.dot(dev, small_omega_inv @ eta))
        t_cdf_q_part2 = np.sqrt((df + dim) / (df + maha))
        z = t_cdf_q_part1 * t_cdf_q_part2
        if cdf_approx:
            # Abramowitz and Stegun approximation
            t_cdf = norm.\
                logcdf(z * (1 - 1 / (4 * df)) / np.sqrt(1 + z * z / (2 * df)),
                       loc=0, scale=1)
        else:
            t_cdf = t.logcdf(z, df + dim, loc=0, scale=1)
        # log skew-t
        return _LOG_2 + t_pdf + t_cdf

    def logpdf(self, x, mean=None, psi=None, cov=1, df=None,
               allow_singular=False, cdf_approx=False):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : ndarray
            Quantiles, with the last axis of `x` denoting the components.
        mean : ndarray
            Mean of the distribution.
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        df : float
            Degree of freedom of the distribution.
        allow_singular : bool, optional
            Whether to allow a singular matrix.  (Default: True)
        cdf_approx : bool, optional
            Whether to compute an approximation of the student cdf.
             (Default: False)

        Returns
        -------
        pdf : ndarray or scalar
            Log of the probability density function evaluated at `x`
        """
        dim, mean, cov, psi, df = self._process_parameters(None, mean,
                                                           psi, cov, df)
        x = self._process_quantiles(x, dim)
        omega = cov + psi @ np.transpose(psi)
        psd = PSD(omega, allow_singular=allow_singular)
        out = self._logpdf(x, mean, psi, cov, psd.U, df,
                           psd.log_pdet, psd.rank, dim, cdf_approx)

        return _squeeze_output(out)

    def pdf(self, x, mean=None, psi=None, cov=1, df=None,
            allow_singular=False, cdf_approx=False):
        """
        Multivariate normal probability density function.

        Parameters
        ----------
        x : ndarray
            Quantiles, with the last axis of `x` denoting the components.
        mean : ndarray
            Mean of the distribution.
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        df : float
            Degree of freedom of the distribution.
        allow_singular : bool, optional
            Whether to allow a singular matrix.  (Default: True)
        cdf_approx : bool, optional
            Whether to compute an approximation of the student cdf.
             (Default: False)

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`
        """
        dim, mean, cov, psi, df = self._process_parameters(None, mean,
                                                           psi, cov, df)
        x = self._process_quantiles(x, dim)
        omega = cov + psi @ np.transpose(psi)
        psd = PSD(omega, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, mean, psi, cov, psd.U, df,
                                  psd.log_pdet, psd.rank, dim, cdf_approx))
        return _squeeze_output(out)

    def _rvs(self, size=1, mean=None, psi=None, cov=1,
             df=None, random_state=None):
        """
        Draw random samples from a multivariate skew-t distribution.
        Parameters
        ----------
        size : integer, optional
            Number of samples to draw (default 1).
        mean : ndarray
            Mean of the distribution.
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        df : float
            Degree of freedom of the distribution.
        allow_singular : bool, optional
            Whether to allow a singular matrix.  (Default: True)
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `dim`), where `dim` is the
            dimension of the random variable.

        Notes
        -----
        As this function does no argument checking, it should not be
        called directly; use 'rvs' instead.

        """
        random_state = self._get_random_state(random_state)
        w = gamma.rvs(a=df / 2, scale=2 / df, size=size,
                      random_state=random_state)
        sqrt_w = np.sqrt(w)
        z = _rtrunc_norm(mean=0, sd=1 / sqrt_w, lower=0, upper=float('inf'),
                         random_state=random_state)
        e = multivariate_normal.rvs(cov=cov, size=size,
                                    random_state=random_state)
        rv_part1 = mean[:, np.newaxis] + np.transpose(e)
        rv_part2 = psi * z[np.newaxis, :] / sqrt_w
        out = rv_part1 + rv_part2
        return _squeeze_output(out)

    def rvs(self, size=1, mean=None, psi=None, cov=1,
            df=None, random_state=None):
        """
        Draw random samples from a multivariate skew-t distribution.
        Parameters
        ----------
        size : integer, optional
            Number of samples to draw (default 1).
        mean : ndarray
            Mean of the distribution.
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        df : float
            Degree of freedom of the distribution.
        allow_singular : bool, optional
            Whether to allow a singular matrix.  (Default: True)

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `dim`), where `dim` is the
            dimension of the random variable.
        """
        dim, mean, cov, psi, df = self._process_parameters(None, mean,
                                                           psi, cov, df)
        out = self._rvs(size, mean, psi, cov, df, random_state)
        return _squeeze_output(out)


multivariate_skew_t = multivariate_skew_t_gen()


class multivariate_skew_t_frozen(multi_rv_frozen):
    def __init__(self, mean=None, psi=None, cov=1, df=None,
                 allow_singular=False, seed=None):
        """
        Create a frozen multivariate skew-t distribution.
        Parameters
        ----------
        mean : ndarray, optional
            Mean of the distribution (default zero)
        psi : ndarray
            Shape of the distribution.
        cov : ndarray
            Covariance matrix of the distribution.
        df : float
            Degree of freedom of the distribution.
        allow_singular : bool, optional
            Whether to allow a singular matrix.  (Default: True)
        allow_singular : bool, optional
            If this flag is True then tolerate a singular
            covariance matrix (default False).
        seed : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            This parameter defines the object to use for drawing random
            variates.
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.

        Examples
        --------

        """
        self._dist = multivariate_skew_t_gen(seed)
        self.dim, self.mean, self.cov, self.psi, self.df = self._dist.\
            process_parameters(None, mean, psi, cov, df)
        self.omega = self.cov + self.psi @ np.transpose(self.psi)
        self.omega_info = PSD(self.omega, allow_singular=allow_singular)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.mean, self.psi, self.omega, self.omega_info.U,
                                 self.df, self.omega_info.log_pdet,
                                 self.omega_info.rank, self.dim)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        return self._dist._rvs(size, self.mean, self.psi,
                               self.cov, self.df, random_state)