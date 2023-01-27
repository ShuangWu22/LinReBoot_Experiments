import numpy as np
#import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats

from BanditEnvironment import Linear_Bandit


def Linear_ReBoot_G(env, lam, weight_sd, coefficient_sharing = True):
    '''
    Gaussian Residual Boostrap Exploration assuming linear contextual bandit
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic bandit environment
        lam: regularization parameter
        weight_sd: standard deviation of residual bootstrap weights
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    '''
    # set up
    n = env.n
    K = env.k
    d = env.d
    lam = lam + 1e-20
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_est = np.zeros(d*K).reshape(d, K)
        V_est = [np.identity(d)*lam for i in range(K)]
        g = [np.zeros(d).reshape(d,1) for i in range(K)]
        arm_count = np.zeros(K)
        Sum1_by_A = np.zeros(K)
        Sum2_by_A = np.zeros(K)
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K
        
        # temporary liat/array
        mu_est = np.zeros(K)
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            c_t = env.get_context(t, all_arm = False)
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)
            ##incremental update
            V_est[a_t - 1] = V_est[a_t - 1] + np.matmul(c_t, c_t.T)
            g[a_t - 1] = g[a_t - 1] + r_t * c_t.reshape(d,1)
            Sum1_by_A[a_t - 1] = Sum1_by_A[a_t - 1] + r_t
            Sum2_by_A[a_t - 1] = Sum2_by_A[a_t - 1] + r_t**2
            arm_count[a_t - 1] = arm_count[a_t - 1] + 1
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
        # ReBoot loop
        for t in range(K+1, n):
            ## LSE update
            X = X_by_A[a_t - 1]
            Y = Y_by_A[a_t - 1]
            beta_est[:,a_t - 1] = (np.linalg.inv(V_est[a_t - 1]) @ g[a_t - 1]).reshape(d)

            
            ## ReBoot exploration
            c_t = env.get_context(t, all_arm = False)
            mu_hat = np.matmul(c_t.T, beta_est).reshape(K)
            Sigma_diag = (Sum2_by_A + arm_count * mu_hat * mu_hat - 2 * mu_hat * Sum1_by_A)/(arm_count*arm_count) 
            #Sigma_diag = (Sum2_by_A + arm_count * mu_hat * mu_hat - 2 * mu_hat * Sum1_by_A)/arm_count
            mu_est = stats.multivariate_normal.rvs(size = 1, mean = mu_hat, cov = np.diag(weight_sd**2 * Sigma_diag))
            
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)
            
            ## incremental update
            V_est[a_t - 1] = V_est[a_t - 1] + np.matmul(c_t, c_t.T)
            g[a_t - 1] = g[a_t - 1] + r_t * c_t.reshape(d,1)
            Sum1_by_A[a_t - 1] = Sum1_by_A[a_t - 1] + r_t
            Sum2_by_A[a_t - 1] = Sum2_by_A[a_t - 1] + r_t**2
            arm_count[a_t - 1] = arm_count[a_t - 1] + 1

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
    else:
        beta_est = np.zeros(d).reshape(d, 1)
        V_est = np.identity(d)*lam
        g = np.zeros(d).reshape(d,1)
        arm_count = np.zeros(K)
        Sum1_by_A = np.zeros(K)
        Sum2_by_A = np.zeros(K)
        Y = np.empty((0,1))
        X = np.empty((0,d))
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            r_t = env.pull(a_t, t)
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            V_est = V_est + np.matmul(c_t, c_t.T)
            g = g + r_t * c_t.reshape(d,1)
            Sum1_by_A[a_t - 1] = Sum1_by_A[a_t - 1] + r_t
            Sum2_by_A[a_t - 1] = Sum2_by_A[a_t - 1] + r_t**2
            arm_count[a_t - 1] = arm_count[a_t - 1] + 1
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
        # ReBoot loop
        for t in range(K+1, n):
            ## LSE update
            beta_est = np.linalg.inv(V_est) @ g
            
            ## ReBoot exploration
            c_K = env.get_context(t, True)
            mu_hat = np.matmul(c_K, beta_est).reshape(K)
            Sigma_diag = (Sum2_by_A + arm_count * mu_hat * mu_hat - 2 * mu_hat * Sum1_by_A)/(arm_count*arm_count) 
            #Sigma_diag = (Sum2_by_A + arm_count * mu_hat * mu_hat - 2 * mu_hat * Sum1_by_A)/arm_count
            mu_est = stats.multivariate_normal.rvs(size = 1, mean = mu_hat, cov = np.diag(weight_sd**2 * Sigma_diag))
            
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            
            # incremental update
            V_est = V_est + np.matmul(c_t, c_t.T)
            g = g + r_t * c_t.reshape(d,1)
            Sum1_by_A[a_t - 1] = Sum1_by_A[a_t - 1] + r_t
            Sum2_by_A[a_t - 1] = Sum2_by_A[a_t - 1] + r_t**2
            arm_count[a_t - 1] = arm_count[a_t - 1] + 1

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
    return R, A, regret


def Linear_TS_G(env, tau, coefficient_sharing = True):
    '''
    Linear Thompson Sampling with Gaussian prior N(0, (tau^2)*I)
    Same piror for each arm
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic linear bandit environment
        tau: std in prior (positive float)
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    ''' 
    # set up
    n = env.n
    K = env.k
    d = env.d
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_mean = [np.zeros(d) for i in range(K)]
        beta_V = [np.identity(d)*(1/tau**2) for i in range(K)]
        beta_sigma = [np.identity(d)*tau**2 for i in range(K)]
        beta_g = [np.zeros(d).reshape(d,1) for i in range(K)]
        sigma_est = np.ones(K)
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K


        # temporary liat/array
        beta_est = np.zeros(K*d).reshape(K,d)

        # Thompson Sampling loop
        for t in range(1, n):
            ## estimation
            c_t = env.get_context(t, all_arm = False)
            for j in range(K):
                beta_est[j,:]  = stats.multivariate_normal.rvs(mean = beta_mean[j].reshape(d), cov = (sigma_est[j]**2)*beta_sigma[j])
            c_t = np.array(c_t).reshape(d, 1)
            mu_est = np.matmul(beta_est, c_t).reshape(K)

            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)

            ## update
            X = X_by_A[a_t - 1]
            Y = Y_by_A[a_t - 1]
            sigma_est[a_t - 1] = np.std(Y)
            beta_V[a_t - 1] = beta_V[a_t - 1] + np.matmul(c_t, c_t.T)
            beta_sigma[a_t - 1] = np.linalg.inv(beta_V[a_t - 1])
            beta_g[a_t - 1] = beta_g[a_t - 1] + r_t * c_t.reshape(d,1)
            beta_mean[a_t - 1] = np.matmul(beta_sigma[a_t - 1], beta_g[a_t - 1])


            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
    
    else:
        beta_mean = np.zeros(d)
        beta_V = np.identity(d)*(1/tau**2)
        beta_sigma = np.identity(d)*tau**2
        beta_g = np.zeros(d).reshape(d,1)
        sigma_est = np.ones(1)
        Y = np.empty((0,1))
        X = np.empty((0,d))

        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            r_t = env.pull(a_t, t)
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            sigma_est = np.std(Y)
            beta_V = beta_V + np.matmul(c_t, c_t.T)
            beta_sigma = np.linalg.inv(beta_V)
            beta_g = beta_g + r_t * c_t.reshape(d,1)
            beta_mean = np.matmul(beta_sigma, beta_g)
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        # Thompson Sampling loop
        for t in range(K+1, n):            
            ## estimation
            c_K = env.get_context(t, True)
            beta_est  = stats.multivariate_normal.rvs(mean = beta_mean.reshape(d), cov = (sigma_est**2)*beta_sigma)
            mu_est = np.matmul(c_K, beta_est).reshape(K)
        
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)

            ## update
            sigma_est = np.std(Y)
            beta_V = beta_V + np.matmul(c_t, c_t.T)
            beta_sigma = np.linalg.inv(beta_V)
            beta_g = beta_g + r_t * c_t.reshape(d,1)
            beta_mean = np.matmul(beta_sigma, beta_g)

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
    return R, A, regret


def Linear_TS_IG(env, tau, alpha, coefficient_sharing = True):
    '''
    Linear Thompson Sampling with priors N(0, (tau^2)*I) and IG(alpha, alpha)
    Same piror for each arm
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic linear bandit environment
        tau: std in Gaussian prior (positive float)
        alpha: parameter in Inverse Gamma prior
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    ''' 
    # set up
    n = env.n
    K = env.k
    d = env.d
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_mean = [np.zeros(d) for i in range(K)]
        beta_V = [np.identity(d)*(1/tau**2) for i in range(K)]
        beta_sigma = [np.identity(d)*tau**2 for i in range(K)]
        beta_g = [np.zeros(d).reshape(d,1) for i in range(K)]
        eta_a = np.ones(K)*alpha
        eta_b = np.ones(K)*alpha
        sigma_est = np.ones(K)
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K


        # temporary liat/array
        beta_est = np.zeros(K*d).reshape(K,d)

        # Thompson Sampling loop
        for t in range(1, n):
            c_t = env.get_context(t, False)
            for j in range(K):
                sigma_est[j] = scipy.stats.invgamma.rvs(a=eta_a[j], scale=eta_b[j])
                beta_est[j,:]  = stats.multivariate_normal.rvs(mean = beta_mean[j].reshape(d), cov = sigma_est[j]*beta_sigma[j])
            c_t = np.array(c_t).reshape(d, 1)
            mu_est = np.matmul(beta_est, c_t).reshape(K)

            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.transpose(c_t), axis = 0)

            ## update
            X = X_by_A[a_t - 1]
            Y = Y_by_A[a_t - 1]
            beta_V[a_t - 1] = beta_V[a_t - 1] + np.matmul(c_t, c_t.T)
            beta_sigma[a_t - 1] = np.linalg.inv(beta_V[a_t - 1])
            beta_g[a_t - 1] = beta_g[a_t - 1] + r_t * c_t.reshape(d,1)
            beta_mean[a_t - 1] = np.matmul(beta_sigma[a_t - 1], beta_g[a_t - 1])
            eta_a[a_t - 1] = eta_a[a_t - 1] + 1/2
            eta_b[a_t - 1] = alpha + 0.5*np.matmul(beta_g[a_t - 1].T, beta_mean[a_t - 1])

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)

    
    else:
        beta_mean = np.zeros(d)
        beta_V = np.identity(d)*(1/tau**2)
        beta_sigma = np.identity(d)*tau**2
        beta_g = np.zeros(d).reshape(d,1)
        eta_a = np.ones(1)*alpha
        eta_b = np.ones(1)*alpha
        sigma_est = np.ones(1)
        Y = np.empty((0,1))
        X = np.empty((0,d))
        X_K = np.zeros(d*K).reshape(K,d)
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            r_t = env.pull(a_t, t)
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            beta_V = beta_V + np.matmul(c_t, c_t.T)
            beta_sigma = np.linalg.inv(beta_V)
            beta_g = beta_g + r_t * c_t.reshape(d,1)
            beta_mean = np.matmul(beta_sigma, beta_g)
            eta_a = eta_a + 1/2
            eta_b = alpha + 0.5*np.matmul(beta_g.T, beta_mean)
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        # Thompson Sampling loop
        for t in range(K+1, n):         
            ## estimation
            c_K = env.get_context(t, True)
            sigma_est = scipy.stats.invgamma.rvs(a=eta_a, scale=eta_b)
            beta_est  = stats.multivariate_normal.rvs(mean = beta_mean.reshape(d), cov = (sigma_est**2)*beta_sigma)
            mu_est = np.matmul(c_K, beta_est).reshape(K)
        
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            
            # update
            beta_V = beta_V + np.matmul(c_t, c_t.T)
            beta_sigma = np.linalg.inv(beta_V)
            beta_g = beta_g + r_t * c_t.reshape(d,1)
            beta_mean = np.matmul(beta_sigma, beta_g)
            eta_a = eta_a + 1/2
            eta_b = alpha + 0.5*np.matmul(beta_g.T, beta_mean)
            
            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
    return R, A, regret


def Linear_GIRO(env, a, lam, R_upper, R_lower, coefficient_sharing = True):
    '''
    Garbage In, Reward Out Algorithm
    Boostrap exploration for bounded reward
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic bandit environment
        a: number of postive/negative pseudo rewards per time unit (int)
        lam: regularization parameter 
        R_upper: upper bound for reward  
        R_lower: lower bound for reward
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    '''
    # set up
    n = env.n
    K = env.k
    d = env.d
    lam = lam + 1e-20
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_est = [np.zeros(d) for i in range(K)]
        V_est = [np.identity(d)*lam for i in range(K)]
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K

        # temporary liat/array
        mu_est = np.zeros(K)

        # GIRO loop
        for t in range(1, n):
            c_t = env.get_context(t, False)
            c_t = np.array(c_t).reshape(d, 1)
            for k in range(K):
                s = len(Y_by_A[k])
                ## Boostrapping
                idx = np.random.choice(s,s)
                X = X_by_A[k][idx,:]
                Y = Y_by_A[k][idx,:]
                V_est[k] = np.linalg.inv(np.matmul(np.transpose(X), X) + np.identity(d)*lam)
                beta_est[k] = np.matmul(V_est[k], np.matmul(np.transpose(X), Y))
                mu_est[k] = np.matmul(np.transpose(beta_est[k]), c_t)

            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.transpose(c_t), axis = 0)

            ## pseudo rewards
            for i in range(a):
                Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(R_upper).reshape(1, 1), axis = 0)
                X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.transpose(c_t), axis = 0)
                Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(R_lower).reshape(1, 1), axis = 0)
                X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.transpose(c_t), axis = 0)

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
    
    else:
        beta_est = np.zeros(d)
        V_est = np.identity(d)*lam
        Y = np.empty((0,1))
        X = np.empty((0,d))
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
        # GIRO loop
        for t in range(K+1, n):
            ## Boostrapping
            s = len(Y)
            idx = np.random.choice(s,s)
            X_boot = X[idx,:]
            Y_boot = Y[idx,:]
            c_K = env.get_context(t, True)
            V_est = np.linalg.inv(np.matmul(np.transpose(X_boot), X_boot) + np.identity(d)*lam)
            beta_est = np.matmul(V_est, np.matmul(np.transpose(X_boot), Y_boot))
            mu_est = np.matmul(c_K, beta_est).reshape(K)
        
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            
            ## pseudo rewards
            for i in range(a):
                Y = np.append(Y, np.array(R_upper).reshape(1, 1), axis = 0)
                X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
                Y = np.append(Y, np.array(R_lower).reshape(1, 1), axis = 0)
                X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            
            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        
    return R, A, regret


def Linear_PHE(env, a, lam, R_upper, R_lower, coefficient_sharing = True):
    '''
    Perturbed-history exploratio Algorithm
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic bandit environment
        a: Perturbation scale
        lam: regularization parameter
        R_upper: upper bound for reward  
        R_lower: lower bound for reward
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    ''' 
    # set up
    n = env.n
    K = env.k
    d = env.d
    lam = lam + 1e-20
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_est = [np.zeros(d) for i in range(K)]
        V_est = [np.identity(d)*lam*(a+1) for i in range(K)]
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K
        arm_count = np.zeros(K)

        # temporary liat/array
        mu_est = np.zeros(K)
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            c_t = env.get_context(t, False)
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)
            V_est[a_t - 1] = V_est[a_t - 1] + (a + 1) * np.matmul(c_t, c_t.T)
            arm_count[a_t - 1] = arm_count[a_t  - 1] + 1
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        # PHE loop
        for t in range(K+1, n):
            ## pertubed Histroy
            c_t = env.get_context(t, False)
            c_t = np.array(c_t).reshape(d, 1)
            U_sum = np.random.binomial(np.ceil(a * (t-1)).astype(int), 1/2)
            U = np.random.multinomial(U_sum, [1.0 / (t-1)]*(t-1), size = 1)
            Z_all = R_lower + (R_upper - R_lower) * U.reshape(t-1,1)
            start = int(0)
            for k in range(K):
                X = X_by_A[k]
                Y = Y_by_A[k]
                s = arm_count[k].astype(int)
                end = start + s
                end = int(end)
                Z = Z_all[start:end,0]
                Z = Z.reshape(s, 1)
                beta_est[k] = np.linalg.inv(V_est[k]) @ X.T @ (Y + Z)
                mu_est[k] = np.matmul(c_t.T, beta_est[k].reshape(d,1))
                start = int(start+arm_count[k])
        
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)
            
            ## update
            V_est[a_t - 1] = V_est[a_t - 1] + (a + 1) * np.matmul(c_t, c_t.T)
            arm_count[a_t - 1] = arm_count[a_t  - 1] + 1
            
            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)        
    else:
        beta_est = np.zeros(d)
        V_est = np.identity(d)*lam*(a + 1)
        Y = np.empty((0,1))
        X = np.empty((0,d))
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            V_est = V_est + (a + 1) * np.matmul(c_t, c_t.T)
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        # PHE loop
        for t in range(K+1, n):
            ## pertubed Histroy
            c_K = env.get_context(t, True)
            U_sum = np.random.binomial(np.ceil(a * (t-1)).astype(int), 1/2)
            U = np.random.multinomial(U_sum, [1.0 / (t-1)]*(t-1), size = 1)
            Z = R_lower + (R_upper - R_lower) * U.reshape(t-1,1)
            beta_est = np.linalg.inv(V_est) @ X.T @ (Y + Z)
            mu_est = np.matmul(c_K, beta_est).reshape(K)
        
            ## pull arm
            a_t = np.argmax(mu_est) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            
            ## update
            V_est = V_est + (a + 1) * np.matmul(c_t, c_t.T)
            
            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
            
            
    return R, A, regret


def Linear_UCB(env, lam, delta = 0.05, coefficient_sharing = True):
    '''
    Upper Confidence Bound Algorithm
    This is a linear bandit algorithm
    ============================================
    INPUTS
        env: stochastic bandit environment
        lam: regularization parameter
        delta: tolarence, 1-delta is confidence level
        coefficient sharing: True if assuming true coefficient is same among arms
    ============================================
    OUPUTS 
        R: reward sequence, list with length n
        A is action sequence, list with length n
        regret: regret sequence, list with length n
    '''
    # set up
    n = env.n
    K = env.k
    d = env.d
    lam = lam + 1e-20
    regret = [0]
    A = []
    R = []
    
    if not coefficient_sharing:
        beta_est = [np.zeros(d) for i in range(K)]
        V_est = [np.identity(d)*lam for i in range(K)]
        Sigma_est = [np.identity(d)*(1/lam) for i in range(K)]
        g = [np.zeros(d).reshape(d,1) for i in range(K)]
        Y_by_A = [np.empty((0,1))]*K
        X_by_A = [np.empty((0,d))]*K

        # temporary liat/array
        ucb = np.zeros(K)
        radius = np.zeros(K)
        beta_norm_max = np.zeros(K)
        V_det = np.zeros(K)

        # pull each arm once
        for t in range(1, K+1):
            c_t = env.get_context(t, False)
            a_t = t
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)
            V_est[a_t - 1] = V_est[a_t - 1] + np.matmul(c_t, c_t.T)
            Sigma_est[a_t - 1] = np.linalg.inv(V_est[a_t - 1])
            g[a_t - 1] = g[a_t - 1] + r_t*c_t.reshape(d,1)
            beta_est[a_t - 1] = Sigma_est[a_t - 1] @ g[a_t - 1]
            V_det[a_t - 1] = np.linalg.det(V_est[a_t - 1])
            beta_norm_tmp = np.linalg.norm(beta_est[a_t - 1])
            if beta_norm_tmp > beta_norm_max[a_t - 1]:
                beta_norm_max[a_t - 1] = beta_norm_tmp
            radius[a_t - 1] = np.sqrt(lam)*beta_norm_max[a_t - 1] + np.sqrt(2*np.log(1/delta) + np.log(V_det[a_t - 1]/lam**d))
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)

        # UCB loop
        for t in range(K+1, n):
            c_t = env.get_context(t, False)
            for k in range(K):
                ucb[k] = c_t.T @ beta_est[k] + radius[k] * np.sqrt(c_t.T @ Sigma_est[k] @ c_t)

            ## pull arm
            a_t = np.argmax(ucb) + 1
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y_by_A[a_t - 1] = np.append(Y_by_A[a_t - 1], np.array(r_t).reshape(1, 1), axis = 0)
            X_by_A[a_t - 1] = np.append(X_by_A[a_t - 1], np.array(c_t).reshape(1, d), axis = 0)

            ## least square update
            V_est[a_t - 1] = V_est[a_t - 1] + np.matmul(c_t, c_t.T)
            Sigma_est[a_t - 1] = np.linalg.inv(V_est[a_t - 1])
            g[a_t - 1] = g[a_t - 1] + r_t*c_t.reshape(d,1)
            beta_est[a_t - 1] = Sigma_est[a_t - 1] @ g[a_t - 1]

            ## confidence ellipsoid update 
            V_det[a_t - 1] = np.linalg.det(V_est[a_t - 1])
            beta_norm_tmp = np.linalg.norm(beta_est[a_t - 1])
            if beta_norm_tmp > beta_norm_max[a_t - 1]:
                beta_norm_max[a_t - 1] = beta_norm_tmp
            radius[a_t - 1] = np.sqrt(lam)*beta_norm_max[a_t - 1] + np.sqrt(2*np.log(1/delta) + np.log(V_det[a_t - 1]/lam**d))

            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t) 
    
    else:
        beta_est = np.zeros(d)
        V_est = np.identity(d)*lam
        Sigma_est = np.identity(d)*(1/lam)
        g = np.zeros(d).reshape(d,1)
        Y = np.empty((0,1))
        X = np.empty((0,d))
        
        # temporary liat/array
        ucb = np.zeros(K)
        radius = np.zeros(1)
        beta_norm_max = np.zeros(1)
        V_det = np.zeros(1)
        
        # pull each arm once
        for t in range(1, K+1):
            a_t = t
            c_K = env.get_context(t, True)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            r_t = env.pull(a_t, t)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)
            V_est = V_est + np.matmul(c_t, c_t.T)
            Sigma_est = np.linalg.inv(V_est)
            g = g + r_t*c_t.reshape(d,1)
            beta_est = Sigma_est @ g
            V_det = np.linalg.det(V_est)
            beta_norm_tmp = np.linalg.norm(beta_est)
            if beta_norm_tmp > beta_norm_max:
                beta_norm_max = beta_norm_tmp
            radius = np.sqrt(lam)*beta_norm_max + np.sqrt(2*np.log(1/delta) + np.log(V_det/lam**d))
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
        # UCB loop
        for t in range(K+1, n):
            c_K = env.get_context(t, True)
            for k in range(K):
                c_k = c_K[k,:].reshape(d,1)
                ucb[k] = c_k.T @ beta_est + radius * np.sqrt(c_k.T @ Sigma_est @ c_k)

            ## pull arm
            a_t = np.argmax(ucb) + 1
            r_t = env.pull(a_t, t)
            c_t = c_K[a_t - 1,:].reshape(d,1)
            A.append(a_t)
            R.append(r_t)
            Y = np.append(Y, np.array(r_t).reshape(1, 1), axis = 0)
            X = np.append(X, np.array(c_t).reshape(1, d), axis = 0)

            ## least square update
            V_est = V_est + np.matmul(c_t, c_t.T)
            Sigma_est = np.linalg.inv(V_est)
            g = g + r_t*c_t.reshape(d,1)
            beta_est = Sigma_est @ g

            ## confidence ellipsoid update 
            V_det = np.linalg.det(V_est)
            beta_norm_tmp = np.linalg.norm(beta_est)
            if beta_norm_tmp > beta_norm_max:
                beta_norm_max = beta_norm_tmp
            radius = np.sqrt(lam)*beta_norm_max + np.sqrt(2*np.log(1/delta) + np.log(V_det/lam**d))
            
            ## compute regret
            regret_t = regret[t - 1] + env.regret(a_t, t)
            regret.append(regret_t)
        
    return R, A, regret       


