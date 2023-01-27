import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats

from BanditEnvironment import Linear_Bandit
from Algorithms import Linear_ReBoot_G, Linear_TS_G, Linear_TS_IG, Linear_GIRO, Linear_UCB, Linear_PHE


########################################################################
# Experiment 3: comparison under Linear Bandit with covrariates
########################################################################


# Overall design for all following experiements 
nsim = 100
#nsim = 3
horizon_len = 10000
k = 10
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

########################################################################
## Experiment 3.1 tuning: Linear ReBoot-G Tuning
## setting: d = 5
## result: 
########################################################################
d = 5
sigma_seq = np.ones(k) * 0.1
#nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
#nu_0 = nu_0/np.linalg.norm(nu_0)
#nu = [nu_0]*k
#nu = [np.zeros(d)]*k

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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 1")
    
    beta = beta_design(k, d)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = False)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_covariates_res/setting_1/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)



########################################################################
## Experiment 3.2 tuning: Linear ReBoot-G Tuning
## setting: d = 10
## result: 
########################################################################
d = 10
sigma_seq = np.ones(k) * 0.1
#nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
#nu_0 = nu_0/np.linalg.norm(nu_0)
#nu = [nu_0]*k
#nu = [np.zeros(d)]*k

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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 2")
    
    beta = beta_design(k, d)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = False)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_covariates_res/setting_2/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)


########################################################################
## Experiment 3.3 tuning: Linear ReBoot-G Tuning
## setting: d = 20
## result: 
########################################################################
d = 20
sigma_seq = np.ones(k) * 0.1
#nu_0 = np.random.uniform(0, 1, 1*d).reshape(d)
#nu_0 = nu_0/np.linalg.norm(nu_0)
#nu = [nu_0]*k
#nu = [np.zeros(d)]*k

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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 3")
    
    beta = beta_design(k, d)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = False)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_covariates_res/setting_3/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)