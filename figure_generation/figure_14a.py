# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import sys
import os
sys.path.append('..')
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

figures_path = 'paper_figures'
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

# Load the .mat files for each model
ef_data = sio.loadmat('results/earthformer/Noise_Injected_Robustness_Test/results.mat')
unet_data = sio.loadmat('results/unet/Noise_Injected_Robustness_Test/results.mat')
nwm_data = sio.loadmat('results/neuralwave/Noise_Injected_Robustness_Test/results.mat')
nwm_dynamic_data = sio.loadmat('results/nwm_dynamic/Noise_Injected_Robustness_Test/results.mat')

# Extract data
trainset_usage = ef_data['id'].flatten()
ef_rmse = ef_data['rmse_spec'].flatten()
ef_r2 = ef_data['r2_spec'].flatten()
unet_rmse = unet_data['rmse_spec'].flatten()
unet_r2 = unet_data['r2_spec'].flatten()
nwm_rmse = nwm_data['rmse_spec'].flatten()
nwm_r2 = nwm_data['r2_spec'].flatten()

nwm_dynamic_rmse = nwm_dynamic_data['rmse_spec'].flatten()
nwm_dynamic_r2 = nwm_dynamic_data['r2_spec'].flatten()

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

fig, axs = plt.subplots(1, 2, figsize=(16.4, 6.58))

N = len(trainset_usage)
x = np.arange(N)
bar_width = 0.2
offset_nwm = -0.3
offset_ef = 0
offset_unet = 0.3

bars_nwm_rmse = axs[1].bar(x + offset_nwm, nwm_rmse, width=bar_width, color='red', label='NWM')
bars_ef_rmse = axs[1].bar(x + offset_ef, ef_rmse, width=bar_width, color='blue', label='EF')
bars_unet_rmse = axs[1].bar(x + offset_unet, unet_rmse, width=bar_width, color='green', label='UNet')
line_nwm_dynamic_rmse = axs[1].plot(x, nwm_dynamic_rmse, color='grey', linestyle='--', marker='o', linewidth=2, label='NWM Dynamic')

axs[1].set_xticks(x)
axs[1].set_xticklabels([f'{val:.1f}' for val in trainset_usage])
# axs[1].set_xlabel('Sample Density', fontsize=18, fontweight='bold')
axs[1].set_xlabel('Samples Usage', fontsize=18, fontweight='bold')
axs[1].set_ylabel('RMSE', fontsize=18, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.7)
for spine in axs[1].spines.values():
    spine.set_linewidth(1.5)

bars_nwm_r2 = axs[0].bar(x + offset_nwm, nwm_r2, width=bar_width, color='red')
bars_ef_r2 = axs[0].bar(x + offset_ef, ef_r2, width=bar_width, color='blue')
bars_unet_r2 = axs[0].bar(x + offset_unet, unet_r2, width=bar_width, color='green')

line_nwm_dynamic_r2 = axs[0].plot(x, nwm_dynamic_r2, color='grey', linestyle='--', marker='o', linewidth=2)

axs[0].set_xticks(x)
axs[0].set_xticklabels([f'{val:.1f}' for val in trainset_usage])
# axs[0].set_xlabel('Sample Density', fontsize=18, fontweight='bold')
axs[0].set_xlabel('Samples Usage', fontsize=18, fontweight='bold')
axs[0].set_ylabel('CC', fontsize=18, fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.7)
for spine in axs[0].spines.values():
    spine.set_linewidth(1.5)
axs[0].set_ylim(0, 1.1)

handles = [bars_nwm_rmse[0], bars_ef_rmse[0], bars_unet_rmse[0], line_nwm_dynamic_rmse[0]]
labels = ['Neural Wave Model', 'Earthformer', 'UNet', 'NWM(Dynamic Module)']

fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=24, frameon=False, columnspacing=3.0)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig(f'{figures_path}/Figure_14a.tiff', dpi=300, format='tiff', bbox_inches='tight')

# plt.show()