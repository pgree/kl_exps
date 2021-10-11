"""
---------------------------------------------------------------------

This module contains the following functions:

kl_gp_reg:
    performs gaussian process regression using kl expansions

eigen_decomp:
    compute the eigendecomposition of an integral operator

eval_eigenfunction:
    evaluate an eigenfunction at one point

check_l2_error:
    check the difference between the effective and true covariance function

gp_reg_naive:
    perform gp regression using a naive implementation

legendre_nodes_weights:
    compute Gaussian nodes and weights on an interval

legendre_eval_exp:
    evaluate a Legendre expansion

legendre_pols:
    evaluate many Legendre polynomials at one point

legendre_mats:
    compute the matrices that convert function tabulation to expansions

kernel_matern32:
    evaluates matern 3/2 kernel

kernel_se:
    evaluates squared-exponential (Gaussian) kernel

---------------------------------------------------------------------

The functions contained in this module are used for performing Gaussian
process regression by using KL-expansion representations
of Gaussian processes.

See https://arxiv.org/abs/2108.05924 for details.

"""

import unittest
import logging
import numpy as np
from scipy.special import legendre as leg
import matplotlib.pyplot as plt



def kl_gp_reg(n, kern, el, a, b, xs, ys, sigma2, check_err=False):
    '''
    evaluate the conditional mean and covariance of a Gaussian process
    regression using KL-expansions. That is, find the mean coefficients
    in the KL-expansion and the covariance of those coefficients

    Parameters
    ----------
    n: int
        the number of discretization nodes
    kern: function
        the covariance kernel (a function) with calling sequence
        `kern(x, y, el)` where `el` is the timescale hyperparameter
    el: float
        timescale of the covariance kernel
    a: float
        the lower bound of the interval on which the Gaussian process is defined
    b: float
        the upper bound of the interval on which the Gaussian process is defined
    xs: array_like
        the independent variable of the data
    ys: array_like
        the dependent variable of the data, ie the observations
    sigma2: float
        the residual variance
    check_err: bool
        when set to True, this function will compute the L2 accuracy of the
        effective covariance kernel compared to the true covariance kernel

    Returns
    -------
    lams: array_like
        the eigenvalues of the integral operator that's discretized
    coefs: array_like
        n x k array with the expansion coefficients in a Legendre basis
        of the KL-expansion basis functions
    coefs_mean: array_like
        arry of length k. the expectation of the coefficients in a
        kl-expansion basis
    coefs_cov: array_like
        k x k array. the covariance of the posterior coefficients
    '''

    # transform xs and el to [-1, 1]
    xs_scaled = -1.0 + 2.0*(xs - a)/(b-a)
    el2_scaled = 2.0*el/(b-a)
    def kern_scaled(x, y):
        return kern(x, y, el2_scaled)

    # number of data points
    nn = np.shape(xs)[0]

    # eigen decomposition of integral operator
    k, lams, _, coefs = eigen_decomp(n, a=-1.0, b=1.0, kern=kern_scaled)
    lams = lams[:k]
    coefs = coefs[:,:k]

    # scale eigenfunctions
    coefs *= np.sqrt(lams)

    # evaluate legendre polynomials at data points for constructing a
    pols = np.zeros((nn, n))
    for i in range(nn):
        pols[i] = legendre_pols(xs_scaled[i], n-1)

    # construct a
    a = np.dot(pols, coefs)

    # eigendecomposition of of ata
    lams_ata, u = np.linalg.eigh(np.dot(a.T, a))

    # compute posterior mean
    aty = np.dot(a.T, ys)
    d_inv = np.diag(1 / (sigma2 + lams_ata))
    coefs_mean = np.dot(u, np.dot(d_inv, np.dot(u.T, aty)))

    # compute posterior covariance
    d_inv = sigma2*d_inv
    coefs_cov = np.dot(u, np.dot(d_inv, u.T))

    # check accuracy by taking L2 difference between the true and effective
    # covariance kernel
    if check_err:
        err1 = check_l2_err(a, b, kern_scaled, coefs, nn=21)
        err2 = check_l2_err(a, b, kern_scaled, coefs, nn=42)
        print(f'L2 error of effective kernel: {err1}')
        print(f'error accuracy: {err2 - err1}')

    return lams, coefs, coefs_mean, coefs_cov


def eval_eigenfunction(coefs, j, a, b, x):
    '''
    evaluate one basis function in the kl-expansion at one point.

    Parameters
    ----------
    coefs: array_like
        an n x m array, each column of which contains the Legendre expansion
        of one eigenfunction.
    j: int
        the index of the eigenfunction to be evaluated
    a: float
        the lower bound of the interval on which the eigenfunction is defined
    b: float
        the upper bound of the interval on which the eigenfunction is defined
    x: float
        the point in \([a, b]\) at which to evaluate the eigenfunction

    Returns
    -------
    f: float
        the value of the eigenfunction
    '''

    n = coefs.shape[0]

    # move x to [a, b]
    x_scaled = -1.0 + 2.0*(x - a)/(b-a)

    # evaluate
    f = legendre_eval_exp(n-1, coefs[:,j], x_scaled)

    return f


def gp_reg_naive(kern, xs, sigma2):
    '''
    construct the full inverse of the covariance matrix that appears in
    Gaussian process regression by taking an eigendecomposition of the
    covariance matrix -- \(K(x_i, x_j) + \sigma^2 * I\). This function
    exists for testing purposes.

    Parameters
    ----------
    kern: function
        the covariance kernel with calling sequence kern(x, y)
    xs: array_like
        the independent variable of the observed data
    sigma2: float
        the residual variance (nugget)

    Returns
    -------
    c_inv: array_like
        n x n array where n is the length of xs and ys. this matrix is
        the inverse of the covariance matrix
    '''

    nn = np.shape(xs)[0]
    gram_mat = np.zeros((nn,nn))

    # construct gram matrix
    for i in range(nn):
        for j in range(nn):
            gram_mat[i, j] = kern(xs[i], xs[j])
    
    # take eigendecomposition
    lams, u = np.linalg.eigh(gram_mat)

    # construct (K + sigma2 * I)^{-1}
    lams_inv = 1/(lams+ sigma2)
    c_inv = np.dot(u, np.dot(np.diag(lams_inv), u.T))
  
    return c_inv


def check_l2_err(a, b, kern, coefs, nn):
    '''check the accruracy of a KL-expansion by computing the L2
    norm of the difference between the true kernel and the
    effective kernel -- the outerproduct of eigenfunctions. That
    \(L^2\) difference is an integral over the square
    \([a, b] \\times [a, b] \subseteq \mathbb{R}^2\) and is computed
    using a tensor product of Gaussian nodes.

    Parameters
    ----------
    a: float
        the lower bound of the interval on which the eigenfunctions are defined
    b: float
        the upper bound of the interval on which the eigenfunctions are defined
    kern: function
        the true covariance kernel with calling sequence kern(x, y)
    coefs: array_like
        n x m array of the Legendre expansions of the basis functions of the
        KL-expansion
    nn: int
        the number of Gaussian nodes in each direction

    Returns
    -------
    err: float
        the error
    '''

    # get length of legendre expansions
    n, k = np.shape(coefs)

    ts, whts = legendre_nodes_weights(a, b, nn)

    err = 0.0
    for i in range(nn):
        for j in range(nn):
            x = ts[i]
            y = ts[j]
            dsum = 0.0
            for ijk in range(k):
                f1 = legendre_eval_exp(n-1, coefs[:,ijk], x)
                f2 = legendre_eval_exp(n-1, coefs[:,ijk], y)
                dsum += f1*f2
            err += (dsum - kern(x, y))**2 * whts[i]*whts[j]
    err = np.sqrt(err)

    return err


def legendre_eval_exp(n, legendre_coefs, x):
    '''
    evaluate a legendre expansion at one point using the three-term recurrence
    forumula

    Parameters
    ----------
    n: int
        the length of the Legendre expansion
    legendre_coefs: array_like
        array containing the coefficients in the Legendre expansion
    x: float
        the point at which to evaluate the Legendre expansion

    Returns
    -------
    f: float
        the value of the expansion
    '''

    # initialize recurrence
    pjm2 = 1
    pjm1 = x

    # first two terms
    f = legendre_coefs[0]*pjm2+legendre_coefs[1]*pjm1

    # recurrence
    for j in range(2, n+1):
        pj=  ((2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
        f=f+legendre_coefs[j]*pj
        pjm2 = pjm1
        pjm1 = pj

    return f


def legendre_mats(n):
    '''
    construct matrices for transforming coefficients in a Legendre 
    expansion to tabulations at Gaussian nodes and vice versa

    Parameters
    ----------
    n: int
        the size of the square matrices to be returned

    Returns
    -------
    coefs_to_vals: array_like
        n x n matrix that transforms coefficients in legendre expansion 
        to its values at Gaussian nodes
    vals_to_coefs: array_like
        n x n matrix that transforms values at Gaussian nodes to 
        Legendre expansions
    '''
    
    # initialize matrices
    vals_to_coefs = np.zeros((n,n))
    coefs_to_vals = np.zeros((n,n))

    # construct order-n Gaussian nodes and weights
    a = -1.0
    b = 1.0
    x, whts = legendre_nodes_weights(a, b, n)

    for i in range(n):
        coefs_to_vals[i] = legendre_pols(x[i], n-1)
 
    # now, v converts coefficients of a legendre expansion
    # into its values at the gaussian nodes. construct its
    # inverse u, converting the values of a function at
    # gaussian nodes into the coefficients of a legendre
    # expansion of that function
    for i in range(n):
        d = 1.0
        d = d * (2.0 * (i+1) - 1.0) / 2.0
        for j in range(n):
            vals_to_coefs[i,j]=coefs_to_vals[j,i]*whts[j]*d

    return vals_to_coefs, coefs_to_vals


def legendre_pols(x, n):
    '''
    Evaluate the Legendre polynomials of order 0,1,2,...,n at one point

    Parameters
    ----------
    n: float
        the point at which to evluate the polynomials
    n: int
        the order of the largest Legendre polynomial to be evaluated

    Returns
    -------
    pols: array_like
        the Legendre polynomials evaluated at one point
    '''

    # initialize recurrence
    pk = 1.0 
    pkp1 = x 

    if n == 0: 
        return np.array([1.0])
    if n == 1: 
        return np.array([1.0, x])

    # n is greater than 2. conduct recursion
    pols = np.zeros(n+1)
    pols[0] = 1.0
    pols[1] = x
    for k in range(1, n):
        pkm1 = pk
        pk = pkp1
        pkp1 = ((2.0*k+1.0)*x*pk-k*pkm1)/(k+1.0)
        pols[k+1] = pkp1        

    return pols
    

def eigen_decomp(n, a, b, kern, tol=10**(-13)):
    """Evaluate the eigendecomposition of the integral operator 

      $$Kf(x) = \int_{a}^{b} k(x, y) f(y) dy$$

    using a Nystrom method. 

    Parameters
    ----------
    n: int
        the number of discretization nodes    
    a: float
        the lower bound of the interval on which the eigenfunctions are defined
    b: float
        the upper bound of the interval on which the eigenfunctions are defined
    kern: function
        the true covariance kernel with calling sequence kern(x, y)
    tol: float
        the number of eigenvalues of magnitude greater than `tol` is 
        returned by this function

    Returns
    -------
    k: int
        the number of eigenvalues with magnitude greater than `tol`
    lams: array_like
        array of length n containing eigenvalues of the integral operator
    vals: array_like
        n x n array with eigenfunctions tabulated at gaussian nodes
    coefs: array_like
        n x n array with legendre expansion of eigenvectors
    """

    xs, whts = legendre_nodes_weights(a, b, n)
    
    # construct kernel matrix
    a = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a[i, j] = kern(xs[i], xs[j]) * np.sqrt(whts[i]*whts[j])
    
    # eigendecomposition of symmetric matrix a
    lams, vals = np.linalg.eigh(a)

    # sort eigenvalues in descending order
    args = np.argsort(lams)[::-1]
    lams = lams[args]
    vals = vals[:,args]
    
    # truncate eigendecomposition 
    for i in range(n):
        if np.abs(lams[i]/lams[0]) > tol:
            k = i

    # scale eigenfunctions back to tabulation space
    for i in range(n):
        for j in range(n):
            vals[i, j] = vals[i, j] / np.sqrt(whts[i])

    # convert eigenfunctions to coefficient space
    vals_to_coefs, _ = legendre_mats(n)
    coefs = np.dot(vals_to_coefs, vals)

    return k, lams, vals, coefs


def legendre_nodes_weights(a, b, n):
    '''
    Compute order-n Gaussian nodes and weights

    Parameters
    ----------
    a: float
        the lower bound of the interval on which the Gaussian nodes are defined
    b: float
        the upper bound of the interval on which the Gaussian nodes are defined
    n: int
        the order of Gaussian nodes

    Returns
    -------
    xs: array_like
        the order-n Gaussian nodes
    weights: array_like
        the order-n Gaussian weights, scaled according to the size of the
        interval specified in the calling sequence
    '''

    # initialize nodes and weights
    xs = np.zeros(n)
    weights = np.zeros(n)

    # get nodes and weights
    xleg = leg(n).weights[:,0]
    wleg = leg(n).weights[:,1]

    # scale weights and scale and shift nodes
    for i in range(n):
        tmp = (xleg[i] + 1.0)/2.0
        xs[i] = a + (b-a)*tmp
        weights[i] = wleg[i]*(b-a)/2.0

    return xs, weights


def kernel_se(x, y, el):
    '''
    squared exponential kernel

    Parameters
    ----------
    x: float
        one argument of the kernel
    y: float
        the other argument of the kernel
    el: float
        the timescale of the kernel

    Returns
    -------
    f: float
        the kernel evaluation
    '''
    return np.exp(-(x - y)**2/(2.0*el**2))


def kernel_matern32(x, y, el):
    '''
    matern 3/2 kernel 

    Parameters
    ----------
    x: float
        one argument of the kernel
    y: float
        the other argument of the kernel
    el: float
        the timescale of the kernel

    Returns
    -------
    f: float
        the kernel evaluation
    '''
    tmp = np.sqrt(3)*np.abs(x-y)/el
    return (1.0 + tmp) * np.exp(-tmp)


class TestKL(unittest.TestCase):
    """For unit testing"""
    def test1(self):
        """in this testing routine, we compare the results of Gaussian
        process regression using KL expansions with the same procedure
        using the straightforward naive algorithm that requires o(n^3)
        operations.
        """
    
        np.random.seed(1)
    
        # set kernel
        el = 0.5
        def kern(x, y):
            return kernel_se(x, y, el)
    
        # set residual variance
        sigma2 = 0.7
        
        # construct data and true solution
        a = -1.0
        b = 1.0
        nn = 10
        xs = np.linspace(a, b, nn)
        eps = np.random.normal(size=nn)
        ys = np.cos(3*np.exp(xs/(b-a))) + eps
        nn_true = 100
        xs_true = np.linspace(a, b, nn_true)
        ys_true = np.cos(3*np.exp(xs_true/(b-a)))
        
        # use kl-expansions to compute conditional mean and covariance
        # of a Gaussian process. the mean and covariance are defined over
        # the space of coefficients in the kl-expansion
        n = 100
        lams, coefs, coefs_mean, coefs_cov = kl_gp_reg(n, kernel_se, el,
            a, b, xs, ys, sigma2, check_err=False)
    
        # set points at which to evaluate conditional mean
        nn_sol = 100
        ts_sol = np.linspace(a, b, nn_sol)
        ys_sol = np.zeros_like(ts_sol)
    
        # tabulate conditional mean
        k = np.shape(lams)[0]
        for i in range(nn_sol):
            f = 0.0
            for j in range(k):
                fj = eval_eigenfunction(coefs, j, a, b, ts_sol[i])
                f = f + fj * coefs_mean[j]
            ys_sol[i] = f
    
        # and plot it
        plt.scatter(xs, ys)
        plt.plot(ts_sol, ys_sol, c='red')
        plt.plot(xs_true, ys_true)
        plt.savefig('mle.png')
    
        # compute variance at the points where mean was tabulated
        fs = np.zeros(k)
        variance = np.zeros(nn_sol)
        for i in range(nn_sol):
            for j in range(k):
                fs[j] = eval_eigenfunction(coefs, j, a, b, ts_sol[i])
            variance[i] = np.dot(fs.T, np.dot(coefs_cov, fs.T))
    
        # now check whether using a naive, o(n^3)
        # straightforward algorithm gets the same solution as the
        # kl-expansion solution.
    
        # construct the vector C^{-1}*y
        c_inv = gp_reg_naive(kern, xs, sigma2)
        k_inv_y = np.dot(c_inv, ys)
      
        # tabulate mean at the same points where kl expansions conditional mean
        # was tabulated
        tmp2 = np.zeros((nn_sol, nn))
        for i in range(nn_sol):
            tmp2[i] = kern(ts_sol[i], xs)
        ys_sol2 = np.dot(tmp2, k_inv_y)
    
        # tabulate variances at each point
        variance2 = np.zeros(nn_sol)
        for i in range(nn_sol):
            variance2[i] = kern(0.0, 0.0) - np.dot(tmp2[i],
                                                   np.dot(c_inv, tmp2[i].T))
        
        # assert that the solutions obtained via kl expansions and
        # via naive implementation agree to several digits
        np.testing.assert_allclose(variance, variance2, atol=1e-10)
        np.testing.assert_allclose(ys_sol, ys_sol2, atol=1e-10)
    

if __name__ == '__main__':
    np.seterr(all='ignore')
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
