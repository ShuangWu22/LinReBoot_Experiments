import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats

from BanditEnvironment import Linear_Bandit
from Algorithms import Linear_ReBoot_G, Linear_TS_G, Linear_TS_IG, Linear_GIRO, Linear_UCB, Linear_PHE


########################################################################
# Experiment 2: comparison under Linear Bandit with Random Context
########################################################################


# Overall design for all following experiements 
nsim = 100
#nsim = 3
horizon_len = 10000
k = 100

########################################################################
## Experiment 1.1 tuning: Linear ReBoot-G Tuning
## setting: d = 5
## result: 
########################################################################
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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 1")
    
    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = True)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_random_res/setting_1/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)



########################################################################
## Experiment 1.2 tuning: Linear ReBoot-G Tuning
## setting: d = 10
## result: 
########################################################################
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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 2")
    
    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = True)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_random_res/setting_2/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)


########################################################################
## Experiment 1.3 tuning: Linear ReBoot-G Tuning
## setting: d = 20
## result: 
########################################################################
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

weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
regret_avg_list_LinReBootG = []
regret_std_list_LinReBootG = []
num_tuning = len(weight_sd_list)
regret_matrix_list = [np.empty((0, horizon_len))]*num_tuning

for i in range(nsim):
    
    print("This is the ", i, "-th repetition of Linear ReBoot-G in setting 3")
    
    beta = np.random.uniform(-0.5, 0.5, 1*d).reshape(1, d)
    beta = beta/np.linalg.norm(beta)
    env = Linear_Bandit(k = k, n = horizon_len, beta = beta, sigma = sigma_seq, random_context = True, gen_context = Gaussian_constrained_context)
    for j in range(num_tuning):
        weight_sd = weight_sd_list[j]
        _, _, regret =  Linear_ReBoot_G(env, lam = 0.1, weight_sd = weight_sd, coefficient_sharing = True)
        regret_matrix_list[j] = np.append(regret_matrix_list[j], np.array(regret).reshape(1,horizon_len), axis = 0)

for l in range(num_tuning):    
    regret_avg = np.mean(regret_matrix_list[l], axis = 0)
    regret_std = np.std(regret_matrix_list[l], axis = 0)/np.sqrt(nsim)
    regret_avg_list_LinReBootG.append(regret_avg)
    regret_std_list_LinReBootG.append(regret_std)

for l in range(num_tuning):
    file_path = "Results/LB_random_res/setting_3/LinReBootG_res_" + str(l) + ".csv"
    pd.DataFrame({'regret_avg':regret_avg_list_LinReBootG[l], 'regret_std':regret_std_list_LinReBootG[l]}).to_csv(file_path, index=None)