import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats

from BanditEnvironment import Linear_Bandit
from Algorithms import Linear_ReBoot_G, Linear_TS_G, Linear_TS_IG, Linear_GIRO, Linear_UCB, Linear_PHE


########################################################################
# Experiment 2: comparison under stochastic linear bandit with random context
########################################################################

# Overall design for all following experiements 
nsim = 100
#nsim = 2
horizon_len = 10000
k = 100
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]

########################################################################
## Experiment 2: comparison under stochastic linear bandit with random context
## setting: d = 5
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = 2, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

# Overall design for all following experiements 
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

regret_avg_list = []
regret_std_list = []
regret_matrix_list = [np.empty((0, horizon_len))]*6

for i in range(nsim):

    print("Start the", i, "-th repetition of setting 1")

    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

    print("This is the ", i, "-th repetition for LinReBoot in setting 1")
    _, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
    regret_matrix_list[0] = np.append(regret_matrix_list[0], np.array(regret_rbg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-G in setting 1")
    _, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
    regret_matrix_list[1] = np.append(regret_matrix_list[1], np.array(regret_tsg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-IG in setting 1")
    _, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
    regret_matrix_list[2] = np.append(regret_matrix_list[2], np.array(regret_tsig).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinGIRO in setting 1")
    _, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[3] = np.append(regret_matrix_list[3], np.array(regret_giro).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinPHE in setting 1")
    _, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[4] = np.append(regret_matrix_list[4], np.array(regret_phe).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinUCB in setting 1")
    _, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
    regret_matrix_list[5] = np.append(regret_matrix_list[5], np.array(regret_ucb).reshape(1,horizon_len) ,axis = 0)

for l in range(len(alg_list)):
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list.append(regret_avg)
    regret_std_list.append(regret_std)

# save file
for l in range(len(alg_list)):
    pd.DataFrame({'regret_avg':regret_avg_list[l], 'regret_std':regret_std_list[l]}).to_csv("Results/LB_random_res/setting_1/" + alg_list[l] + ".csv", index=None)



########################################################################
## Experiment 2: comparison under stochastic linear bandit with random context
## setting: d = 10
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = ?, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

# Overall design for all following experiements 
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

regret_avg_list = []
regret_std_list = []
regret_matrix_list = [np.empty((0, horizon_len))]*6

for i in range(nsim):

    print("Start the", i, "-th repetition of setting 2")

    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

    print("This is the ", i, "-th repetition for LinReBoot in setting 2")
    _, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
    regret_matrix_list[0] = np.append(regret_matrix_list[0], np.array(regret_rbg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-G in setting 2")
    _, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
    regret_matrix_list[1] = np.append(regret_matrix_list[1], np.array(regret_tsg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-IG in setting 2")
    _, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
    regret_matrix_list[2] = np.append(regret_matrix_list[2], np.array(regret_tsig).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinGIRO in setting 2")
    _, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[3] = np.append(regret_matrix_list[3], np.array(regret_giro).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinPHE in setting 2")
    _, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[4] = np.append(regret_matrix_list[4], np.array(regret_phe).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinUCB in setting 2")
    _, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
    regret_matrix_list[5] = np.append(regret_matrix_list[5], np.array(regret_ucb).reshape(1,horizon_len) ,axis = 0)

for l in range(len(alg_list)):
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list.append(regret_avg)
    regret_std_list.append(regret_std)

# save file
for l in range(len(alg_list)):
    pd.DataFrame({'regret_avg':regret_avg_list[l], 'regret_std':regret_std_list[l]}).to_csv("Results/LB_random_res/setting_2/" + alg_list[l] + ".csv", index=None)



########################################################################
## Experiment 2: comparison under stochastic linear bandit with random context
## setting: d = 20
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = ?, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

# Overall design for all following experiements 
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

regret_avg_list = []
regret_std_list = []
regret_matrix_list = [np.empty((0, horizon_len))]*6

for i in range(nsim):

    print("Start the", i, "-th repetition of setting 3")

    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)

    print("This is the ", i, "-th repetition for LinReBoot in setting 3")
    _, _, regret_rbg =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.05, coefficient_sharing = True)
    regret_matrix_list[0] = np.append(regret_matrix_list[0], np.array(regret_rbg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-G in setting 3")
    _, _, regret_tsg =  Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
    regret_matrix_list[1] = np.append(regret_matrix_list[1], np.array(regret_tsg).reshape(1,horizon_len) ,axis = 0)

    print("This is the ", i, "-th repetition for LinTS-IG in setting 3")
    _, _, regret_tsig =  Linear_TS_IG(env, tau = np.sqrt(5), alpha = 2, coefficient_sharing = True)
    regret_matrix_list[2] = np.append(regret_matrix_list[2], np.array(regret_tsig).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinGIRO in setting 3")
    _, _, regret_giro =  Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[3] = np.append(regret_matrix_list[3], np.array(regret_giro).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinPHE in setting 3")
    _, _, regret_phe =  Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/2), R_lower = -1 - 3*np.sqrt(1/2), coefficient_sharing = True)
    regret_matrix_list[4] = np.append(regret_matrix_list[4], np.array(regret_phe).reshape(1,horizon_len) ,axis = 0)
    
    print("This is the ", i, "-th repetition for LinUCB in setting 3")
    _, _, regret_ucb =  Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
    regret_matrix_list[5] = np.append(regret_matrix_list[5], np.array(regret_ucb).reshape(1,horizon_len) ,axis = 0)

for l in range(len(alg_list)):
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list.append(regret_avg)
    regret_std_list.append(regret_std)

# save file
for l in range(len(alg_list)):
    pd.DataFrame({'regret_avg':regret_avg_list[l], 'regret_std':regret_std_list[l]}).to_csv("Results/LB_random_res/setting_3/" + alg_list[l] + ".csv", index=None)

