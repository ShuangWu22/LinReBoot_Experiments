import numpy as np
#import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats

class Linear_Bandit():
    '''
    stochastic linear (contextual) bandit environment
    rewards are Gaussian
    ============================================
    k: number of arms (int)
    n: length of horizon (int)
    sigma: reward is sigma^2-SubGaussian, list of positive float with length k
    beta: linear coefficient for mean reward for each arm, true parameter for linear bandit
          k by d(or 1 by d) numpy array for user-defined values.
    random_context: True if contexts are stochastic/generated from some distribution,
                    False if contexts are deterministic/fixed.
    gen_context: function to generate d-dimensional context for all arms, output k by d numpy array
    '''
    def  __init__(self, k, n, beta, sigma, random_context = True, gen_context = None):
        
        self.k = k                                  # number of arms
        self.n = n                                  # length of horizon
        self.sigma = sigma                          # SubGaussian constants
        self.beta = np.array(beta)                  # linear coefficient, parameter for linear bandit
        self.d = beta.shape[1]                      # dimension of context                 
        
        if gen_context is None:
            print("error: please specify a funtion for generating context")
            
        # make tables
        if beta.shape[0] == 1:
            ## This is the case that true coefficient is shared by arms
            betas = np.repeat(beta, k, axis = 0)
        else:
            ## This is the case that coefficients are different among arms(linear bandit with covariates)
            betas = beta
        
        random_arms = np.random.randint(1, k+1, n) 
        d = beta.shape[1]
        reward_table = np.zeros(k*n).reshape(n ,k)
        context_table = np.zeros(d*n*k).reshape(n, k, d)
        mu_table = np.zeros(k*n).reshape(n, k)
        if random_context:
            ## random context, context is generated at each time
            for i in range(n):
                c = gen_context(k = k, d = d)
                context_table[i,:,:] = c
                if beta.shape[0] == 1:
                    for j in range(k):
                        mu_table[i,j] = np.inner(betas[j,:], c[j,:])
                    reward_table[i,:] = stats.multivariate_normal.rvs(mean = mu_table[i,:], cov = np.diag(sigma**2))
                else:
                    c_t = c[random_arms[i] - 1,:]
                    mu_table[i,:] = np.matmul(betas, c_t.reshape(d,1)).reshape(k)
                    reward_table[i,:] = stats.multivariate_normal.rvs(mean = mu_table[i,:], cov = np.diag(sigma**2))
        else:
            ## fixed context, context is fixed along time
            c = gen_context(k = k, d = d)
            mu = np.zeros(k).reshape(k)
            for j in range(k):
                mu[j] = np.inner(betas[j,:], c[j,:])
            for i in range(n):
                context_table[i,:,:] = c
                mu_table[i,:] = mu
                reward_table[i,:] = stats.multivariate_normal.rvs(mean = mu, cov = np.diag(sigma**2))
        
        self.rewards = reward_table                       # reward table, n by k array
        self.contexts = context_table                     # context table, n by k by d array
        self.mus = mu_table                               # mu table, n by k array
        self.random_arms = random_arms                    # random arms sequence with length n for arm-independent context   

    def pull(self, a, t):
        '''
        pull arm/take action and observe reward
        ============================================
        INPUT
            a: action
            t: time
        ============================================
        OUPUT
            r: reward
        '''
        r = self.rewards[t,a-1]
        return r
        
    
    def get_context(self, t, all_arm = True):
        '''
        get context
        ============================================
        INPUT
            t: time
            all_arm: True return k by d array for k contexts; False return one d by 1 context randomly
        ============================================
        OUPUT
            c: context, k by d or d by 1
        '''
        if not all_arm:
            c = self.contexts[t, self.random_arms[t]-1, :].reshape(self.d, 1) 
        else:
            c = self.contexts[t, :, :]
        return c
    
    def regret(self, a, t):
        '''
        regret for current action
        ============================================
        INPUT
            a: action
            t: time
        ============================================
        OUPUT
            regret: regret
        '''
        mu = self.mus[t,:]
        mu_best = max(mu)
        regret = mu_best - mu[a-1]
        return regret
    
    def beta_sharing(self):
        '''
        indicator for beta sharing among arms
        ============================================
        INPUT
        ============================================
        OUPUT
            sharing: True if beta is same among arms
        '''
        return self.beta.shape[0] == 1