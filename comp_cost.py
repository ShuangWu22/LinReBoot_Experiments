import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
import time

from BanditEnvironment import Linear_Bandit
from Algorithms import Linear_ReBoot_G, Linear_TS_G, Linear_TS_IG, Linear_GIRO, Linear_UCB, Linear_PHE


########################################################################
# Experiment 1: comparison under stochastic linear bandit
########################################################################
def Uniform_constrained_context(k, d, min = 0, max = 1):
    '''
    function to generate d-diemsnional context from uniform distribution such that
    norms are 1
    ============================================
    INPUT
        k: number of arms
        d: dimension of context, defualt is 2d context
        min: lower bound for uniform distribution, default is 0
        max: upper bound for uniform distribution, default is 1
    ============================================
    OUPUT
        c: contexts for all arms, k by d numpy array
    '''
    c = np.zeros(k*d).reshape(k,d)
    for i in range(k):
        c_k = np.random.uniform(low = min, high = max, size = d).reshape(d) 
        norm = np.linalg.norm(c_k)
        c_k = c_k / norm
        c[i,:] = c_k
    return c

# Overall design for all following experiements 
nsim = 100
#nsim = 3
horizon_len = 10000
k = 100
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]

## d =5
d = 5

print("#################Computational cost for SLB d=5#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
sigma_seq = np.ones(k) * np.sqrt(0.1)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = False, gen_context = Uniform_constrained_context)


time_start = time.perf_counter()
Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.3, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)


## d = 10
d = 10

print("#################Computational cost for SLB d=10#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
sigma_seq = np.ones(k) * np.sqrt(0.1)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = False, gen_context = Uniform_constrained_context)


time_start = time.perf_counter()
Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.3, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)

## d = 20
d = 20

print("#################Computational cost for SLB d=20#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
sigma_seq = np.ones(k) * np.sqrt(0.1)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = False, gen_context = Uniform_constrained_context)


time_start = time.perf_counter()
Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.3, coefficient_sharing = True)
time_elapsed = (time.perf_counter()- time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)



########################################################################
# Experiment 2: comparison under stochastic linear bandit with random context
########################################################################

# Overall design for all following experiements 
nsim = 100
#nsim = 2
horizon_len = 10000
k = 100
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]

# d=5
d = 5
sigma_seq = np.ones(k) * np.sqrt(0.5)
nu = []
for kk in range(k):
    nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
    nu_0 = nu_0/np.linalg.norm(nu_0)
    nu = nu + [nu_0]

def Gaussian_constrained_context(k, d, mean = nu):
    '''
    function to generate d-diemsnional context from Gaussian distribution
    such that norms are 1
    ============================================
    INPUT
        k: number of arms
        d: dimension of context, defualt is 2d context
        mean: mean for d-dimensional Gasussian
    ============================================
    OUPUT
        c: contexts for all arms, k by d numpy array
    '''
    cov = [1/(2*k)*np.identity(d)]*k
    c = np.zeros(k*d).reshape(k,d)
    for i in range(k):
        c_k = np.array(stats.multivariate_normal.rvs(mean = mean[i], cov = cov[i], size = 1)).reshape(d) 
        norm = np.linalg.norm(c_k)
        c_k = c_k / norm
        c[i,:] = c_k
    return c

print("#################Computational cost for LB Random d=5#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)



# d=10
d = 10
sigma_seq = np.ones(k) * np.sqrt(0.5)
nu = []
for kk in range(k):
    nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
    nu_0 = nu_0/np.linalg.norm(nu_0)
    nu = nu + [nu_0]

def Gaussian_constrained_context(k, d, mean = nu):
    '''
    function to generate d-diemsnional context from Gaussian distribution
    such that norms are 1
    ============================================
    INPUT
        k: number of arms
        d: dimension of context, defualt is 2d context
        mean: mean for d-dimensional Gasussian
    ============================================
    OUPUT
        c: contexts for all arms, k by d numpy array
    '''
    cov = [1/(2*k)*np.identity(d)]*k
    c = np.zeros(k*d).reshape(k,d)
    for i in range(k):
        c_k = np.array(stats.multivariate_normal.rvs(mean = mean[i], cov = cov[i], size = 1)).reshape(d) 
        norm = np.linalg.norm(c_k)
        c_k = c_k / norm
        c[i,:] = c_k
    return c

print("#################Computational cost for LB Random d=10#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)


# d = 20
d = 20
sigma_seq = np.ones(k) * np.sqrt(0.5)
nu = []
for kk in range(k):
    nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
    nu_0 = nu_0/np.linalg.norm(nu_0)
    nu = nu + [nu_0]

def Gaussian_constrained_context(k, d, mean = nu):
    '''
    function to generate d-diemsnional context from Gaussian distribution
    such that norms are 1
    ============================================
    INPUT
        k: number of arms
        d: dimension of context, defualt is 2d context
        mean: mean for d-dimensional Gasussian
    ============================================
    OUPUT
        c: contexts for all arms, k by d numpy array
    '''
    cov = [1/(2*k)*np.identity(d)]*k
    c = np.zeros(k*d).reshape(k,d)
    for i in range(k):
        c_k = np.array(stats.multivariate_normal.rvs(mean = mean[i], cov = cov[i], size = 1)).reshape(d) 
        norm = np.linalg.norm(c_k)
        c_k = c_k / norm
        c[i,:] = c_k
    return c

print("#################Computational cost for LB Random d=20#################")

beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
beta = beta/np.linalg.norm(beta)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)





########################################################################
# Experiment 3: comparison under Linear Bandit with covrariates
########################################################################

# Overall design for all following experiements 
nsim = 100
#nsim = 3
horizon_len = 10000
k = 10
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]

def beta_design(k, d):
    '''
    function to beta for linear bandit with covariates setting
    ============================================
    INPUT
        k: number of arms
        d: dimension of context
    ============================================
    OUPUT
        beta: true parameter, k by d
    '''
    beta = np.ones(k*d).reshape(k,d)
    idx = []
    for kk in range(k):
        b = np.random.binomial(d, 1/2)
        idx_k = np.random.choice(d, b, replace = False)
        beta[kk,:][idx_k] = -1
        beta[kk,:] = beta[kk,:] + np.random.uniform(low = -0.95, high = 0.95, size = d)
        beta[kk,:] = (kk + 1)/k * beta[kk,:]/np.linalg.norm(beta[kk,:])
    
    return beta

def Gaussian_constrained_context(k, d):
    '''
    function to generate d-diemsnional context from Gaussian distribution
    such that norms are 1
    ============================================
    INPUT
        k: number of arms
        d: dimension of context, defualt is 2d context
        mean: mean for d-dimensional Gasussian
    ============================================
    OUPUT
        c: contexts for all arms, k by d numpy array
    '''
    #cov = [np.identity(d)]*k
    c = np.zeros(k*d).reshape(k,d)
    for i in range(k):
        c_k = np.array(stats.multivariate_normal.rvs(mean = np.zeros(d), cov = np.identity(d), size = 1)).reshape(d) 
        norm = np.linalg.norm(c_k)
        c_k = c_k / norm
        c[i,:] = c_k
    return c




# d = 5 
d = 5
sigma_seq = np.ones(k) * 0.1

print("#################Computational cost for LB Covariates d=5#################")

beta = beta_design(k, d)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 1, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(10), alpha = 2, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)

# d = 10
d = 10
sigma_seq = np.ones(k) * 0.1

print("#################Computational cost for LB Covariates d=10#################")


beta = beta_design(k, d)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 1, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(10), alpha = 2, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)


# d = 20
d = 20
sigma_seq = np.ones(k) * 0.1

print("#################Computational cost for LB Covariates d=20#################")


beta = beta_design(k, d)
env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

time_start = time.perf_counter()
_, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 1, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[0], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[1], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(10), alpha = 2, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[2], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[3], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1.3, R_lower = -1.3, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[4], " is:", "%5.1f secs" % time_elapsed)

time_start = time.perf_counter()
_, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = False)
time_elapsed = (time.perf_counter() - time_start)
print("Computational time for", alg_list[5], " is:", "%5.1f secs" % time_elapsed)