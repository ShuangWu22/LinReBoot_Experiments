import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



weight_sd_list = [0.05, 0.1, 0.2, 0.5, 1]
color_list = ['b', 'c', 'g', 'k', 'm', 'y']
num_tuning = len(weight_sd_list)
horizon_len = 10000
# plots
plt.clf()
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

########################################################################
## Experiment 2.1 tuning: Linear ReBoot-G Tuning
## setting: d = 5
## result: 
########################################################################

for i in range(num_tuning):
    file = "Results/LB_random_res/setting_1/LinReBootG_res_" + str(i) + ".csv"
    label = "LinReBoot-G(" + str(weight_sd_list[i]) + ")"
    res = pd.read_csv(file)
    ax1.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax1.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax1.legend(bbox_to_anchor=(0.33, 1))
ax1.set_xlabel("Decision Time")
ax1.set_ylabel("Regret")
ax1.set_title("Regret versus T: d=5")


########################################################################
## Experiment 2.2 tuning: Linear ReBoot-G Tuning
## setting: d = 10
## result: 
########################################################################


for i in range(num_tuning):
    file = "Results/LB_random_res/setting_2/LinReBootG_res_" + str(i) + ".csv"
    label = "LinReBoot-G(" + str(weight_sd_list[i]) + ")"
    res = pd.read_csv(file)
    ax2.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax2.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax2.legend(bbox_to_anchor=(0.33, 1))
ax2.set_xlabel("Decision Time")
ax2.set_ylabel("Regret")
ax2.set_title("Regret versus T: d= 10")


########################################################################
## Experiment 2.3 tuning: Linear ReBoot-G Tuning
## setting: d = 20
## result: 
########################################################################

for i in range(num_tuning):
    file = "Results/LB_random_res/setting_3/LinReBootG_res_" + str(i) + ".csv"
    label = "LinReBoot-G(" + str(weight_sd_list[i]) + ")"
    res = pd.read_csv(file)
    ax3.plot(np.arange(horizon_len), res['regret_avg'], label=label, color = color_list[i])
    ax3.fill_between(np.arange(horizon_len),
                    (res['regret_avg']-res['regret_std']),
                    (res['regret_avg']+res['regret_std']),
                    color = color_list[i], alpha = 0.2)

ax3.legend(bbox_to_anchor=(0.33, 1))
ax3.set_xlabel("Decision Time")
ax3.set_ylabel("Regret")
ax3.set_title("Regret versus T: d=20")

fig.set_size_inches(30, 10)
#plt.show()
fig.savefig('Results/LB_random_res/LB_random_tuning_plot_summary.png', dpi=250)