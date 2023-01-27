import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


color_list = ['b', 'c', 'g', 'k', 'm', 'y']
horizon_len = 10000
# plots
plt.clf()
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]
label_list = ["Linear ReBoot-G", "Linear TS-G", "Linear TS-IG", "Linear GIRO", "Linear PHE", "Linear UCB"]

########################################################################
## Experiment 3: comparison under linear bandit with covariates
## setting: d = 5
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.4, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = ?, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_1/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax1.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax1.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax1.legend(bbox_to_anchor=(0.33, 1))
ax1.set_xlabel("Decision Time")
ax1.set_ylabel("Regret")
ax1.set_title("Regret versus T: LB covariates under d=5")

########################################################################
## Experiment 3: comparison under linear bandit with covariates
## setting: d = 10
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.4, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = ?, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_2/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax2.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax2.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax2.legend(bbox_to_anchor=(0.33, 1))
ax2.set_xlabel("Decision Time")
ax2.set_ylabel("Regret")
ax2.set_title("Regret versus T: LB covariates under d=10")

########################################################################
## Experiment 3: comparison under linear bandit with covariates
## setting: d = 20
## Linear ReBoot-G    Linear_ReBoot_G(env, lam = 0.1, weight_sd = 0.5, coefficient_sharing = True)
## Linear TS-G        Linear_TS_G(env, tau = np.sqrt(10), coefficient_sharing = True)
## Linear TS-IG       Linear_TS_IG(env, tau = np.sqrt(10), alpha = ?, coefficient_sharing = True)
## Linear GIRO        Linear_GIRO(env, a = 1, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear PHE         Linear_PHE(env, a = 0.5, lam = 0.1, R_upper = 1 + 3*np.sqrt(1/10), R_lower = -1 - 3*np.sqrt(1/10), coefficient_sharing = True)
## Linear UCB         Linear_UCB(env, lam = 0.1, delta = 0.05, coefficient_sharing = True)
########################################################################

for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_3/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax3.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax3.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax3.legend(bbox_to_anchor=(0.33, 1))
ax3.set_xlabel("Decision Time")
ax3.set_ylabel("Regret")
ax3.set_title("Regret versus T: LB covariates under d=20")

fig.set_size_inches(30, 10)
plt.show()
fig.savefig('Results/LB_covariates_res/LB_covariates_Exp_plot.png', dpi=250)