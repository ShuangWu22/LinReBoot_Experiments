import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import stats
import time

color_list = ['b', 'c', 'g', 'k', 'm', 'y']
horizon_len = 10000
# plots
plt.clf()
fig, ax = plt.subplots(3,3)
alg_list = ["LinReBoot-G", "LinTS-G", "LinTS-IG", "LinGIRO", "LinPHE", "LinUCB"]
label_list = ["Linear ReBoot", "Linear TS-G", "Linear TS-IG", "Linear GIRO", "Linear PHE", "Linear UCB"]

## SLB d=5
for i in range(len(alg_list)):
    file = "Results/SLB_res/setting_1/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[0,0].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[0,0].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[0,0].legend(bbox_to_anchor=(0.33, 1))
#ax[0,0].set_xlabel("Decision Time", fontsize=15)
ax[0,0].set_ylabel("Regret in Setting 1", fontsize=28)
ax[0,0].tick_params(axis = "both", labelsize =25)
ax[0,0].set_ylim(0, 600)
ax[0,0].set_title("d=5", fontsize=32)


### SLB d=10
for i in range(len(alg_list)):
    file = "Results/SLB_res/setting_2/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[0,1].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[0,1].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[0,1].legend(bbox_to_anchor=(0.33, 1))
#ax[0,1].set_xlabel("Decision Time", fontsize=15)
#ax[0,1].set_ylabel("Regret", fontsize=15)
ax[0,1].tick_params(axis = "both", labelsize =25)
ax[0,1].set_ylim(0, 600)
ax[0,1].set_title("d=10", fontsize=32)


### SLB d=20
for i in range(len(alg_list)):
    file = "Results/SLB_res/setting_3/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[0,2].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[0,2].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[0,2].legend(bbox_to_anchor=(0.33, 1))
#ax[0,2].set_xlabel("Decision Time", fontsize=15)
#ax[0,2].set_ylabel("Regret", fontsize=15)
ax[0,2].tick_params(axis = "both", labelsize =25)
ax[0,2].set_ylim(0, 600)
ax[0,2].set_title("d=20", fontsize=32)



### LB random d=5
for i in range(len(alg_list)):
    file = "Results/LB_random_res/setting_1/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[1,0].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[1,0].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[1,0].legend(bbox_to_anchor=(0.33, 1))
#ax[1,0].set_xlabel("Decision Time", fontsize=15)
ax[1,0].set_ylabel("Regret in Setting 2", fontsize=28)
ax[1,0].tick_params(axis = "both", labelsize =25)
ax[1,0].set_ylim(0, 800)
#ax[1,0].set_title("Linear Bandit with Random Context: d=5", fontsize=22)


### LB random d=10
for i in range(len(alg_list)):
    file = "Results/LB_random_res/setting_2/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[1,1].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[1,1].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[1,1].legend(bbox_to_anchor=(0.33, 1))
#ax[1,1].set_xlabel("Decision Time", fontsize=15)
#ax[1,1].set_ylabel("Regret", fontsize=15)
ax[1,1].tick_params(axis = "both", labelsize =25)
ax[1,1].set_ylim(0, 800)
#ax[1,1].set_title("Linear Bandit with Random Context: d=10", fontsize=22)


### LB random d =20
for i in range(len(alg_list)):
    file = "Results/LB_random_res/setting_3/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[1,2].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[1,2].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[1,2].legend(bbox_to_anchor=(0.33, 1))
#ax[1,2].set_xlabel("Decision Time", fontsize=15)
#ax[1,2].set_ylabel("Regret", fontsize=15)
ax[1,2].tick_params(axis = "both", labelsize =25)
ax[1,2].set_ylim(0, 800)
#ax[1,2].set_title("Linear Bandit with Random Context: d=20", fontsize=22)


### LB covariates d=5
for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_1/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[2,0].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[2,0].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[2,0].legend(bbox_to_anchor=(0.33, 1))
ax[2,0].set_xlabel("Decision Time", fontsize=28)
ax[2,0].set_ylabel("Regret in Setting 3", fontsize=28)
ax[2,0].tick_params(axis = "both", labelsize =25)
ax[2,0].set_ylim(0, 1000)
#ax[2,0].set_title("Linear Bandit with Covariates: d=5", fontsize=22)


### LB covariates d=10
for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_2/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[2,1].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[2,1].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

#ax[2,1].legend(bbox_to_anchor=(0.33, 1))
ax[2,1].set_xlabel("Decision Time", fontsize=28)
#ax[2,1].set_ylabel("Regret", fontsize=15)
ax[2,1].tick_params(axis = "both", labelsize =25)
ax[2,1].set_ylim(0, 1000)
#ax[2,1].set_title("Linear Bandit with Covariates: d=10", fontsize=22)


### LB covariates d=20
for i in range(len(alg_list)):
    file = "Results/LB_covariates_res/setting_3/" + alg_list[i] + ".csv"
    label = label_list[i]
    res = pd.read_csv(file)
    ax[2,2].plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i], linewidth=2)
    ax[2,2].fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax[2,2].legend(loc='upper center', bbox_to_anchor=(-0.75, -0.22), prop={'size': 30}, ncol=6)
ax[2,2].set_xlabel("Decision Time", fontsize=28)
#ax[2,2].set_ylabel("Regret", fontsize=15)
ax[2,2].tick_params(axis = "both", labelsize =25)
ax[2,2].set_ylim(0, 1000)
#ax[2,2].set_title("Linear Bandit with Covariates: d=20", fontsize=22)



fig.set_size_inches(34, 20)
plt.show()
fig.savefig('Results/Summary.png', dpi=200)