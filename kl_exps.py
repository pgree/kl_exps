import numpy as np
from scipy.special import legendre as leg
import matplotlib.pyplot as plt
import unittest
import logging


def eval_eigenfunction(coefs, j, t0, t1, x):
    '''
    evaluate one basis function in the kl-expansion at one point.

    Parameters
    ----------
    coefs: array_like
        an n x m np.array. Each column contains the Legendre expansion
        of one eigenfunction. 
    j: int
        the index of the eigenfunction to be evaluated
    t0: float
        the lower bound of the interval on which the eigenfunction is defined
    t1: float
        the upper bound of the interval on which the eigenfunction is defined
    x: float
        the point in \([t0, t1]\) at which to evaluate the eigenfunction

    Returns
    -------
    f: float
        the value of the eigenfunction 

    '''

    n = coefs.shape[0]

    # move x to [t0, t1]
    x2 = -1.0 + 2.0*(x - t0)/(t1-t0)

    # evaluate
    f = lege_eval_exp(n-1, coefs[:,j], x2)

    return f



def gp_reg_naive(kern, xs, ys, sigma2):
    '''
    construct the full inverse of the covariance matrix that appears in 
    Gaussian process regression by taking an eigendecomposition of the 
    covariance matrix. this function exists for testing purposes. 

    Parameters
    ----------
    kern: function
        the covariance kernel with calling sequence kern(x, y)
    xs: array_like
        the independent variable of the observed data
    ys: array_like
        the dependent variables of the observed data
    sigma2: float
        the residual variance (nugget)

    Returns
    -------
    c_inv: array_like
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



def kl_gp_reg(n, kern, el, t0, t1, xs, ys, sigma2, check_err=False):
    '''
    evaluate the conditional mean and covariance of a gaussian process using
    KL-expansions. That is, find the mean coefficients in the KL-expansion and 
    the covariance of those coefficients 

    Parameters
    ----------
    n: int
        the number of discretization nodes
    kern: function
        the covariance kernel (a function)
    xs: array_like
        the independent variable of the data
    ys: array_like
        the dependent variable of the data, ie the observations
    sigma2: float
        the residual variance 

    Returns
    -------
    lams: array_like
        the eigenvalues of the integral operator that's discretized
    coefs: array_like
        the expansion coefficients of the eigenfunctions in a Legendre basis
    coefs_mean: array_like
        the posterior mean, the expectation of the coefficients in a 
        kl-expansion basis
    coefs_cov: array_like
        the covariance of the posterior coefficients
    '''

    # transform xs and el to [-1, 1]
    xs2 = -1.0 + 2.0*(xs - t0)/(t1-t0)
    el2 = 2.0*el/(t1-t0)
    def kern2(x, y): return kern(x, y, el2)

    # number of data points
    nn = np.shape(xs)[0]

    # eigen decomposition of integral operator
    t0 = -1.0
    t1 = 1.0
    k, lams, u, coefs = eigen_decomp(n, t0, t1, kern2)
    lams = lams[:k]
    coefs = coefs[:,:k]

    # scale eigenfunctions
    for i in range(n):
        for j in range(k):
            coefs[i, j] = coefs[i, j] * np.sqrt(lams[j])

    # evaluate legendre polynomials at data points for constructing a
    pols = np.zeros((nn, n))
    for i in range(nn):
        pols[i] = lege_pols(xs2[i], n-1)

    # construct a
    a = np.dot(pols, coefs)
    
    # eigendecomposition of of ata 
    lams_ata, u = np.linalg.eigh(np.dot(a.T, a))
    ###print(np.dot(a.T, a))

    # compute posterior mean 
    aty = np.dot(a.T, ys)
    d_inv = 1 / (sigma2 + lams_ata)
    d_inv = np.diag(d_inv)
    coefs_mean = np.dot(u, np.dot(d_inv, np.dot(u.T, aty)))

    # compute posterior covariance 
    d_inv = sigma2*d_inv
    coefs_cov = np.dot(u, np.dot(d_inv, u.T))

    # check accuracy by taking L2 difference between the true and effective 
    # covariance kernel
    if check_err:
        nn1 = 21
        err1 = check_l2_err(t0, t1, kern2, coefs, nn1)
        nn2 = 2*nn1
        err2 = check_l2_err(t0, t1, kern2, coefs, nn2)
        print(f'L2 error of effective kernel: {err1}')
        print(f'error accuracy: {err2 - err1}')

    return lams, coefs, coefs_mean, coefs_cov


def check_l2_err(t0, t1, kern, coefs, nn):
    '''
    check the accruracy of a KL-expansion by computing the L2
    norm of the difference between the true kernel and the 
    effective kernel -- the outerproduct of eigenfunctions. That 
    \(L^2\) difference is an integral over the square 
    \([t0, t1] \\times [t0, t1] \subseteq \mathbb{R}^2\) and is computed
    using a tensor product of Gaussian nodes.

    Parameters
    ----------
    t0: float
        the lower bound of the interval on which the eigenfunctions are defined
    t1: float
        the upper bound of the interval on which the eigenfunctions are defined
    kern: function
        the true covariance kernel with calling sequence kern(x, y)
    coefs: array_like
        the Legendre expansions of the eigenfunctions of the KL-expansion
    nn: int
        the number of Gaussian nodes in each direction

    Returns
    -------
    err: float
        the error
    '''

    # get length of legendre expansions
    n, k = np.shape(coefs)

    ts, whts = lege_nodes_weights(t0, t1, nn)

    err = 0.0
    for i in range(nn):
        for j in range(nn):
            x = ts[i]
            y = ts[j]
            dsum = 0.0
            for ijk in range(k):
                f1 = lege_eval_exp(n-1, coefs[:,ijk], x)
                f2 = lege_eval_exp(n-1, coefs[:,ijk], y)
                dsum += f1*f2
            err += (dsum - kern(x, y))**2 * whts[i]*whts[j]
    err = np.sqrt(err)

    return err


def lege_eval_exp(n, lege_coefs, x):
    '''
    evaluate a legendre expansion at one point using the three-term recurrence 
    forumula

    Parameters
    ----------
    n: int
        the length of the Legendre expansion
    lege_coefs: array_like
        the coefficients in the Legendre expansion
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
    f = lege_coefs[0]*pjm2+lege_coefs[1]*pjm1
    der = lege_coefs[1]

    # recurrence 
    for j in range(2, n+1):
        pj=  ((2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
        f=f+lege_coefs[j]*pj
        pjm2 = pjm1
        pjm1 = pj

    return f


def lege_uv(n):
    '''
    construct matrices for transforming coefficients in a Legendre 
    expansion to tabulations at Gaussian nodes and vice versa

    Parameters
    ----------
    n: int
        the size of the square matrices to be returned

    Returns
    -------
    v: array_like
        transforms coefficients in legendre expansion to its values at 
        Gaussian nodes
    u: array_like
        transforms values at Gaussian nodes to Legendre expansions
    '''
    
    # initialize matrices
    u = np.zeros((n,n))
    v = np.zeros((n,n))

    # construct order-n Gaussian nodes and weights
    t0 = -1.0
    t1 = 1.0
    x, whts = lege_nodes_weights(t0, t1, n)

    for i in range(n):
        v[i] = lege_pols(x[i], n-1)
 
    # now, v converts coefficients of a legendre expansion
    # into its values at the gaussian nodes. construct its
    # inverse u, converting the values of a function at
    # gaussian nodes into the coefficients of a legendre
    # expansion of that function
    for i in range(n):
        d = 1.0
        d = d * (2.0 * (i+1) - 1.0) / 2.0
        for j in range(n):
            u[i,j]=v[j,i]*whts[j]*d

    return u, v


def lege_pols(x, n):
    '''
    Evaluate the Legendre polynomials of order 0,1,2,...,n at one point

    Parameters
    ----------
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

    if (n == 0): return np.array([1.0])
    if (n == 1): return np.array([1.0, x])

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
    

def eigen_decomp(n, t0, t1, kern, tol=10**(-13)):
    """Evaluate the eigendecomposition of the integral operator 

      $$Kf(x) = \int_{t0}^{t1} k(x, y) f(y) dy$$

    using a Nystrom method. 

    Parameters
    ----------
    n: int
        the number of discretization nodes    
    t0: float
        the lower bound of the interval on which the eigenfunctions are defined
    t1: float
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
        eigenvalues of the integral operator
    u: array_like
        eigenfunctions tabulated at gaussian nodes
    coefs: array_like
        legendre expansion of eigenvectors
    """

    xs, whts = lege_nodes_weights(t0, t1, n)
    
    # construct kernel matrix
    a = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a[i, j] = kern(xs[i], xs[j])
            a[i, j] *= np.sqrt(whts[i]*whts[j])
    
    # eigendecomposition of symmetric matrix a
    lams, u = np.linalg.eigh(a)

    # sort eigenvalues in descending order
    args = np.argsort(lams)[::-1]
    lams = lams[args]
    u = u[:,args]
    
    # truncate eigendecomposition 
    for i in range(n):
        tmp = np.abs(lams[i]/lams[0])
        if tmp > tol:
            k = i

    # scale eigenfunctions back to tabulation space
    for i in range(n):
        for j in range(n):
            u[i, j] = u[i, j]/np.sqrt(whts[i])

    # convert eigenfunctions to coefficient space
    uleg, vleg = lege_uv(n)
    coefs = np.dot(uleg, u)

    return k, lams, u, coefs


def lege_nodes_weights(t0, t1, n):
    '''
    Compute order-n Gaussian nodes and weights

    Parameters
    ----------
    n: int
        the order of Gaussian nodes
    t0: float
        the lower bound of the interval on which the Gaussian nodes are defined
    t1: float
        the upper bound of the interval on which the Gaussian nodes are defined

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
        xs[i] = t0 + (t1-t0)*tmp
        weights[i] = wleg[i]*(t1-t0)/2.0

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
    tmp = np.sqrt(3)*np.abs(x-y)/el
    return (1.0 + tmp) * np.exp(-tmp)


class TestKL(unittest.TestCase):
    def test1(self):
    
        np.random.seed(1)
    
        # set kernel
        el = 0.5
        def kern(x, y): return kernel_se(x, y, el)
    
        # set residual variance 
        sigma2 = 0.7
        
        # construct data and true solution
        t0 = -1.0
        t1 = 1.0
        nn = 10
        xs = np.linspace(t0, t1, nn)
        eps = np.random.normal(size=nn)
        ys = np.cos(3*np.exp(xs/(t1-t0))) + eps
        nn_true = 100
        xs_true = np.linspace(t0, t1, nn_true)
        ys_true = np.cos(3*np.exp(xs_true/(t1-t0)))
        
        # use kl expansions compute conditional mean and covariance of coefficients 
        n = 100
        lams, coefs, coefs_mean, coefs_cov = kl_gp_reg(n, kernel_se, el, 
            t0, t1, xs, ys, sigma2, check_err=False)
    
        # set points at which to evaluate conditional mean
        nn_sol = 100
        ts_sol = np.linspace(t0, t1, nn_sol)
        ys_sol = np.zeros_like(ts_sol)
    
        # tabulate conditional mean
        k = np.shape(lams)[0]
        for i in range(nn_sol):
            f = 0.0
            for j in range(k):
                fj = eval_eigenfunction(coefs, j, t0, t1, ts_sol[i])
                f = f + fj * coefs_mean[j]
            ys_sol[i] = f
    
        # and plot it
        plt.scatter(xs, ys)
        plt.plot(ts_sol, ys_sol, c='red')
        plt.plot(xs_true, ys_true)
        plt.savefig('mle.png')
    
        # compute variance at the points where mean was tabulated
        fs = np.zeros(k)
        vars = np.zeros(nn_sol)
        for i in range(nn_sol):
            for j in range(k):
                fs[j] = eval_eigenfunction(coefs, j, t0, t1, ts_sol[i])
            vars[i] = np.dot(fs.T, np.dot(coefs_cov, fs.T))
    
    
        # now check whether using a naive, o(n^3)
        # straightforward algorithm gets the same solution as the 
        # kl-expansion solution. 
    
        # construct the vector C^{-1}*y
        c_inv = gp_reg_naive(kern, xs, ys, sigma2)
        k_inv_y = np.dot(c_inv, ys)
      
        # tabulate mean at the same points where kl expansions conditional mean
        # was tabulated 
        tmp2 = np.zeros((nn_sol, nn))
        for i in range(nn_sol):
            tmp2[i] = kern(ts_sol[i], xs)
        ys_sol2 = np.dot(tmp2, k_inv_y)
    
        # tabulate variances at each point
        vars2 = np.zeros(nn_sol)
        for i in range(nn_sol):
            vars2[i] = kern(0.0, 0.0) - np.dot(tmp2[i], np.dot(c_inv, tmp2[i].T))
        
        # assert that the solutions obtained via kl expansions and 
        # via naive implementation agree to several digits
        np.testing.assert_allclose(vars, vars2, atol=1e-10)
        np.testing.assert_allclose(ys_sol, ys_sol2, atol=1e-10)
    

if __name__ == '__main__':
    np.seterr(all='ignore')
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    unittest.main()
