import numpy as np
import kl_exps as kl
import matplotlib.pyplot as plt
np.seterr(all='ignore')


def main():
    np.random.seed(1)
    
    # set kernel and residual variance
    el = 2.0
    sigma2 = 1.0
        
    # construct data and true solution
    t0 = -10.0
    t1 = 1.0
    nn = 100
    xs = np.sort(np.random.uniform(t0, t1, size=nn))
    eps = np.random.normal(size=nn)
    ys = np.cos(3*np.exp(xs/(t1-t0))) + eps
    nn_true = 100
    xs_true = np.linspace(t0, t1, nn_true)
    ys_true = np.cos(3*np.exp(xs_true/(t1-t0)))
        
    # use kl expansions compute conditional mean and covariance of coefficients 
    n = 100
    lams, coefs, coefs_mean, coefs_cov = kl.kl_gp_reg(n, kl.kernel_matern32, 
        el, t0, t1, xs, ys, sigma2, check_err=False)
    print(f'conditional mean of coefficients:\n {coefs_mean}')
    print(coefs_cov)
    plt.imshow(np.log(np.abs(coefs_cov)))
    plt.show()

    # set points at which to evaluate conditional mean
    nn_sol = 100
    ts_sol = np.linspace(t0, t1, nn_sol)
    ys_sol = np.zeros_like(ts_sol)
    
    # tabulate conditional mean
    k = np.shape(lams)[0]
    for i in range(nn_sol):
        f = 0.0
        for j in range(k):
            fj = kl.eval_eigenfunction(coefs, j, t0, t1, ts_sol[i])
            f = f + fj * coefs_mean[j]
        ys_sol[i] = f
    
    # and plot it
    plt.scatter(xs, ys)
    plt.plot(ts_sol, ys_sol, c='red')
    plt.plot(xs_true, ys_true)
    plt.savefig('mle.png')
    

if __name__ == '__main__':
    main()
